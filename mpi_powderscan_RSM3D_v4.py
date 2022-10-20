''' 
Script to analyze powder diffraction data 

Originally written by Christian Schleputz
Modified for current version by Zhan Zhang, APS, ANL. 

This version utilizes the now internal package "PowderScanMapper"
Note at this time, the plot function is not implemented. 

The generated xye file will be in a sub-folder with the same name as the 
  corresponding SPEC file. 

Change list:

    2022/10/13 (ZZ):
       - merged the 2.1 and 2.0.3 version together
       - this one has both the realtime function and the slice removal option
       - recent modification includes:
            - directory making on Windows platform.  (used to work fine on Linux)
            - output filename with indicator of _q vs. _tth.  
           
    2022/10/14 (ZZ):
       - try to check the scan is done or not on realtime option with the 
           EPICS PV flag.  Just implemented at 33-ID-D.  
    
    2022/10/15 (ZZ):
       - Get the json input added.
       - change the scan number list handling to be the same as the RSM one.
    
To DO:
    
    - SPEC file reindex check file time stamp? No need to redo if not changed. 
    - MPI version?
    - how to handle like every other scans? 
'''
#================================================
# Import some useful generic packages. 
import os
import sys
import time
import datetime
from pathlib import Path
import numpy as np
import json
import argparse
from spec2nexus import spec
from itertools import zip_longest

#=====================================
#  Import RsMap3D package and setup update progress
import rsMap3D
from rsMap3D.datasource.Sector33SpecDataSource import Sector33SpecDataSource
from rsMap3D.datasource.DetectorGeometryForXrayutilitiesReader import DetectorGeometryForXrayutilitiesReader as detReader
from rsMap3D.utils.srange import srange
from rsMap3D.config.rsmap3dconfigparser import RSMap3DConfigParser
from rsMap3D.transforms.unitytransform3d import UnityTransform3D
#from rsMap3D.mappers.gridmapper import QGridMapper
from rsMap3D.mappers.powderscanmapper import PowderScanMapper
from rsMap3D.mappers.output.powderscanwriter import PowderScanWriter
from epics import PV
#=====================================

def updateDataSourceProgress(value1, value2):
    logger.info("\t\tDataLoading Progress %.3f%%/%s%%" % (value1, value2))

def updateMapperProgress(value1):
    logger.info("\t\tMapper Progress -- Current Curve %.3f%%" % (value1))

def reindex_specfile(fullFilename):
    logger.info('=============================')
    logger.info('  Re-indexing the SPEC file...')
    spec_data = spec.SpecDataFile(fullFilename)
    scansInFile_list = spec_data.getScanNumbers()
    logger.info('  Done. The lastest scan # is %s' % scansInFile_list[-1])
    return scansInFile_list, spec_data
    
def parseArgs():
    parser = argparse.ArgumentParser(description='Default MPI run script. Expects a json config file pointed at the data.')
    parser.add_argument('configPath', 
                        nargs='?', 
                        default=os.path.join(os.getcwd(), 'pdconfig.json'),
                        help='Path to config file. If none supplied, directs to a config.json located in CWD')
    return parser.parse_args()

def generateScanLists(inputScanList):
    # How many conditions/cycles in total
    num_cycles = inputScanList["cycles"]
    # Total number of scans at one condition, say each temperature
    scans_in_1_cycle = inputScanList["scans_per_cycle"]
    SetsOfRSM = inputScanList["rsm_sets"]
    
    scanListTop = []
    for i in range(0, num_cycles):
        for oneSetRSM in SetsOfRSM:
            scan_s = oneSetRSM["start"]
            scan_e = oneSetRSM["end"]
            scans_in_1_rsm = scan_e - scan_s + 1
            scanListTop = scanListTop + \
                ( [[f"{x}" if scans_in_1_rsm==1 else f"{x}-{x+scans_in_1_rsm-1}"] for \
                x in range(scan_s+i*scans_in_1_cycle, scan_e+i*scans_in_1_cycle+1, scans_in_1_rsm)] )
    return scanListTop

def as_list(listInput):
    # generate the scan_list from a number/string/range
    listOutput = listInput
    if not isinstance(listInput, list):
        if isinstance(listInput, (int, float)):
            listOutput = [int(listInput)]
        elif isinstance(listInput, str):
            listOutput = srange(listInput).list()
        elif isinstance(listInput, range):
            listOutput = list(listInput)
    return listOutput

# This one checks the PV set by SPEC, indicating the SPEC file,
#   the scan number, and scan status.  Return 1 if the select scan 
#   has been finished or not. 
def _is_scan_done(specfile_full, curr_scan=1):
    specfile_pv = PV("33idSIS:spec:SPECFileName")
    scann_pv = PV("33idSIS:spec:SCANNUM")
    scanDone_pv = PV("33idSIS:spec:ISSCANDONE")
    
    # Check if PVs are there.  If one cant be reached, stop there. 
    if (not scann_pv.connect(timeout=2)) \
        or (not scanDone_pv.connect(timeout=2)) \
        or (not specfile_pv.connect(timeout=2)):
        return -1
    
    # If it is not the current file, this one is irrelavent.
    if specfile_full != specfile_pv.char_value:
        return -1
    # If scan number is lower than the current, go ahead
    if (curr_scan < scann_pv.value):
        return 1
    elif (curr_scan == scann_pv.value):
        if scanDone_pv.value == 1:
            return 1
        else:
            return 0
    else:
        return 0

# Rocord the starting run time
startTime = datetime.datetime.now()
with open('time.log', 'a') as time_log:
    time_log.write(f'Start: {startTime}\n')

#================================================
# try use the logger to do console display
import logging

# create root_logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
# create formatter
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(levelname)s - %(message)s')
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
root_logger.addHandler(ch)

# create logger
logger = logging.getLogger('powderScan_RSM3D')
logger.setLevel(logging.INFO)
#================================================

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Get the input from the input file
args = parseArgs()
with open(args.configPath, 'r') as config_f:
    config = json.load(config_f)

# workpath and config file path
projectDir = config["project_dir"]
configDir = config["config_dir"]

if configDir == None:
    configDir = projectDir
    
# Config files 
detectorConfigName = os.path.join(configDir, config["detector_config"])
instConfigName = os.path.join(configDir, config["instrument_config"])
badPixelFile = os.path.join(configDir, config["badpixel_file"])
flatfieldFile = config["flat_field"]
                      
# Detector settngs
detectorName = config["detector_name"]
bin = config["binning"]
roi = config["roi_setting"]
# Do realtime or not
realtime_flag = config["real_time"]
# Plot/save or not.  
do_plot = config["do_plot"]
plot_y = config["plot_y"]
write_file = config["write_file"]
# Output selction
data_coordinate = config["x-axis"]
x_min_0 = config["x_min"]
x_max_0 = config["x_max"]
x_step = config["x_step"]

datasets = config["datasets"]

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#####################################
# checking for validity of the inputs
if not os.path.exists(detectorConfigName):
    raise Exception("Detector Config file does not exist: %s" % 
                    detectorConfigName)
if not os.path.exists(instConfigName):
    raise Exception("Instrument Config file does not exist: %s" % 
                    instConfigName)
if not (badPixelFile is None):
    if not os.path.exists(badPixelFile):
        raise Exception("Bad Pixel file does not exist: %s" % 
                        badPixelFile)
if not (flatfieldFile is None):
    flatfieldFile = os.path.join(configDir, flatfieldFile)
    if not os.path.exists(flatfieldFile):
        raise Exception("Flat field file does not exist: %s" % 
                        flatfieldFile)
    
# Set ROI or get info from the config
if roi is None:
    detector = dReader.getDetectorById(detectorName)
    nPixels = dReader.getNpixels(detector)
    roi = [1, nPixels[0], 1, nPixels[1]]
logger.info('  ROI: %s ' % roi)
# End checking for inputs
#####################################

dReader = detReader(detectorConfigName)

#=====================================
# Outer-most loop, iterate over different SPEC files
for idx, dataset in enumerate(datasets, 1):
    # SPEC file name and scan numbers
    specFile = dataset["spec_file"]
    scanListTop = dataset["scan_list"]
    slicesNotUsed_list = dataset["slice_not_used"]
    if scanListTop == None:
        inputScanList = dataset["scan_range"]
        scanListTop = generateScanLists(inputScanList)
    
    specfile_full = os.path.join(projectDir, specFile)
    # Check if the SPEC file exists. Quit if not. 
    if not os.path.exists( specfile_full ):
        print(f"File not found.  Please check the path and name {specFile} is correct.")
        print(f"Moving on...in a couple of seconds. ")
        time.sleep(2)
        continue

    specName, specExt = os.path.splitext(specFile)
    # generate the output file folder now
    outputFilePath = os.path.join(projectDir, \
            "analysis_runtime", specName)
    Path(outputFilePath).mkdir(parents=True, exist_ok=True)

    logger.info('=============================')
    logger.info(f"SPEC file #{idx}: {specfile_full}")
    logger.info(f"Generated Scan List #{idx}: {scanListTop}")

    # generate the scan_list from a number/string/range -- not sure if this is needed
    scanListTop = as_list(scanListTop)
            
    # generate the slicesNotUsed_list from the given list/strings
    slicesNotUsed_scanN = []
    slicesNotUsed_sliceN = []
    for slicesNotUsed_oneset in slicesNotUsed_list:
        scan_range = srange(slicesNotUsed_oneset[0])
        scan_num = scan_range.list()
        slice_range = srange(slicesNotUsed_oneset[1])
        slice_nums = slice_range.list()
        if slice_nums:
            slicesNotUsed_scanN.extend( scan_num )
            slicesNotUsed_sliceN.append( slice_nums * len(scan_num) )

    num_curves = len(scanListTop)
    progress = 0
    # (Re)index the SPEC file
    scansInFile_list, spec_data = reindex_specfile(specfile_full)

    # Inner loop here, iterate over set of scans, one curve/file per set
    for idx2, scanList1 in enumerate(scanListTop, 1):
        # Generate the actual list from the input string list. 
        scanRange = []
        for scans in scanList1:
            scanRange += srange(scans).list()
        logger.info(f"  --------------------------------")
        logger.info(f" Set # {idx2} scanRange: {scanRange}")
        #logger.info("specName, specExt: %s, %s" % (specName, specExt))
        
        _n_scans = len(scanRange)
        if not _n_scans:
            continue
            
        # Generate the output filename with full path.
        outputFileName = os.path.join(outputFilePath, \
            f"{specName}_S{scanRange[0]:03}" + \
            "" if _n_scans == 1 else f"-S{scanRange[-1]:03}")
        if data_coordinate == "tth":
            outputFileName += "_tth.xye"
        else:
            outputFileName += "_q.xye"
       
        #=======================
        # This section checks if the scans are available yet.
        # Find the largest scan number in this set
        curr_scan = max(scanRange)
        # waiting time factor
        _accu = 1  
        last_len = 0
        while (realtime_flag):
            # add my local way to check if scan is done
            _scanDone_flag = _is_scan_done(specfile_full, curr_scan)
            if _scanDone_flag == 1:
                break
            elif _scanDone_flag == 0:
                sleep_time = _accu*5
                logger.info('  Scan #%d not done yet.  \
                    Wait %d seconds' % (curr_scan, sleep_time))
            else:
                # check if the largest scan number is available yet.
                if not (str(curr_scan) in scansInFile_list):
                    sleep_time = _accu*5
                    logger.info('  Scan #%d not available yet.  \
                        Wait %d seconds' % (curr_scan, sleep_time))
                elif scansInFile_list.index(str(curr_scan)) == (len(scansInFile_list) - 1):
                    # Try to figure out the scan is done or not -- no good way as I see it.
                    scan = spec_data.getScan(curr_scan)
                    #scan.interpret()
                    if(len(scan.data)>0):
                        curr_len = len(scan.data[scan.L[0]])
                    else:
                        curr_len = 0
                    logger.info('   Data points current in the scan #%d is %d' \
                        % (curr_scan, curr_len) )
                    if(curr_len == 0):
                        pass
                    elif(curr_len != last_len):
                        last_len = curr_len
                    else:
                        if(_accu>4):
                            break
                    sleep_time = 5*_accu
                    logger.info('  Scan #%d may not be finished.  Wait %d seconds...' \
                      % (curr_scan, sleep_time) )
                else:
                    logger.info('\t--------------------------------')
                    logger.info('  Scan #%d ready.\n' % curr_scan)
                    break

                scansInFile_list, spec_data = \
                    reindex_specfile(specfile_full)
                    
            _accu += 1        
            # this is in the outer loop, when the scan does not exist yet, wait sometime before resume loops
            time.sleep(sleep_time)
        #=======================

        # Reset x_min and x_max for of each interation.
        x_min = x_min_0
        x_max = x_max_0

        _start_time = time.time()

        # Initialize the data source from the spec file, detector and instrument
        # configuration, read the data, and set the ranges such that all images
        # will be used.
        appConfig = RSMap3DConfigParser()
        ds = Sector33SpecDataSource(projectDir, specName, specExt, \
                instConfigName, detectorConfigName, roi=roi, pixelsToAverage=bin, \
                scanList = scanRange, badPixelFile = badPixelFile, \
                flatFieldFile = flatfieldFile, appConfig=appConfig)
        ds.setCurrentDetector(detectorName)
        ds.setProgressUpdater(updateDataSourceProgress)
        ds.loadSource()
        ds.setRangeBounds(ds.getOverallRanges())
        logger.info('  --------------------------------')
        
        # Adding here the section to handle the slices that are not going to be used
        #  in each scan -- in working progress, ZZ 2020/03/20
        for onescan in scanRange:
            if onescan in slicesNotUsed_scanN:
                myindex = slicesNotUsed_scanN.index(onescan)
                slice_list = slicesNotUsed_sliceN[myindex]
                logger.info('      Points #%s in scan #%s ignored.  ' % \
                        (str(slice_list), str(onescan)) )
                my_len = len(ds.imageToBeUsed[onescan])
                for slice in slice_list:
                    if slice in range(-my_len, my_len):
                        ds.imageToBeUsed[onescan][slice] = False
            else:
                logger.info('       No slice removed. ')
        logger.info('  --------------------------------')
        
        # calling powder scan Mapper here.
        #  Note here the plot part does not do anything yet.
        powderMapper = PowderScanMapper(ds,
                 outputFileName,
                 transform = UnityTransform3D(),
                 gridWriter = PowderScanWriter(),
                 appConfig = appConfig,
                 dataCoord = data_coordinate,
                 xCoordMin = x_min,
                 xCoordMax = x_max,
                 xCoordStep = x_step,
                 plotResults = do_plot,
                 yScaling  = plot_y,
                 writeXyeFile = write_file)
                
        powderMapper.setProgressUpdater(updateMapperProgress)
        powderMapper.doMap()
        
        # Some mapping information on screen output
        x_min_output = powderMapper.getXCoordMin()
        x_max_output = powderMapper.getXCoordMax()
        nbins = np.round((x_max_output - x_min_output) / x_step)
        logger.info('  \t %s minimum : %.3f' % (data_coordinate, x_min_output))
        logger.info('  \t %s maximum : %.3f' % (data_coordinate, x_max_output))
        logger.info('  \t %s stepsize: %.3f' % (data_coordinate, x_step))
        logger.info('  \t %s nbins   : %.3f' % (data_coordinate, nbins))

        progress += 100.0/num_curves
        #if verbose:
            #print 'Current File Progress: %.1f' % (progress)
            #print 'Elapsed time for Q-conversion: %.1f seconds' % (time.time() - _start_time)
        logger.info('  Elapsed time for current curve: %.3f seconds' % (time.time() - _start_time) )
        logger.info('  Mapper Progress -- Current File : %.1f%%' % (progress) )
        if write_file:
            logger.info('  Output filename : %s' % (outputFileName) )
        else:
            logger.info('  Output file disabled. ')
            
    with open('time.log', 'a') as time_log:
        endTime = datetime.datetime.now()
        time_log.write(f'End: {endTime}\n')
        time_log.write(f'Diff: {endTime - startTime}\n')
            
    logger.info('  --------------------------------')
logger.info('=============================')
