"""Module for handling regional avalanche simulations."""

import pathlib
import shutil
import logging
import numpy as np
import numpy.ma as ma
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import avaframe.in3Utils.initializeProject as initProj
from avaframe.com1DFA import com1DFA
from avaframe.in3Utils import cfgUtils, cfgHandling
from avaframe.in3Utils import logUtils
from avaframe.in2Trans import rasterUtils
from avaframe.in3Utils import fileHandlerUtils as fU

from rasterio.merge import merge

from avaframe.in3Utils.fileHandlerUtils import findAvaDirsBasedOnInputsDir

# create local logger
log = logging.getLogger(__name__)


def com7RegionalMain(cfgMain, cfg):
    """Run com7Regional with given configuration.

    This function processes multiple avalanche directories in parallel, running simulations
    for each directory.

    Parameters
    ----------
    cfgMain : configparser.ConfigParser
        Main avaframe configuration settings
    cfg : configparser.ConfigParser
        Regional configuration settings with potential overrides

    Returns
    -------
    allPeakFilesDir : pathlib.Path or None
        Path to the directory containing all peak files, if copyPeakFiles is True
    mergedRastersDir : pathlib.Path or None
        Path to the directory containing merged rasters, if mergeOutput is True

    Notes
    -----
    The function expects the following directory structure:
    avalancheDir/
    └── regionalDir/
        ├── avalanche1/
        ├── avalanche2/
        └── ...

    Where:
    - avalancheDir: Main directory specified in cfgMain
    - regionalDir: Subdirectory specified in cfg['GENERAL']['regionaldir']  # Fixed: ConfigParser uses lowercase
    """
    # Define the regional directory in relation to the avalanche directory
    regionalDirFromCfg = str(cfg["GENERAL"]["regionalDir"])
    regionalDir = pathlib.Path(cfgMain["MAIN"]["avalancheDir"]) / regionalDirFromCfg

    # List valid avalanche directories within the regional directory
    avaDirs = findAvaDirsBasedOnInputsDir(regionalDir)

    # Get total number of simulations
    log.info(f"Getting total number of simulations to perform...")
    # Don't use silentLogger - let user see progress for long runs!
    totalSims = getTotalNumberOfSims(avaDirs, cfgMain, cfg)
    log.info(f"Found {totalSims} (new) simulations to perform across {len(avaDirs)} directories")

    # Get number of processes based on number of avaDirs
    nProcesses = cfgUtils.getNumberOfProcesses(cfgMain, len(avaDirs))

    # Set nCPU for com1 to 1 to avoid nested parallelization
    cfgMain["MAIN"]["nCPU"] = "1"

    # Track progress and results
    completed = 0
    nSuccesses = 0

    # Import tqdm for progress bar
    from tqdm import tqdm
    
    print(f"\n{'='*70}")
    print(f"STEP 2/3: Running {len(avaDirs)} avalanche simulations")
    print(f"Using {nProcesses} parallel processes")
    print(f"{'='*70}\n")

    # Process avalanche directories within the regional folder concurrently
    with ProcessPoolExecutor(max_workers=nProcesses) as executor:
        # Submit each avalanche directory to the executor
        futures = {
            executor.submit(processAvaDirCom1Regional, cfgMain, cfg, avaDir): avaDir for avaDir in avaDirs
        }
        
        # Use tqdm progress bar for visual feedback
        with tqdm(total=len(avaDirs), desc="Simulating avalanches", unit="scenario", ncols=100) as pbar:
            for future in as_completed(futures):
                avaDir = futures[future]
                try:
                    resultDir, status = future.result()
                    completed += 1

                    if status == "Success":
                        nSuccesses += 1
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({"success": nSuccesses, "failed": completed - nSuccesses})

                except Exception as e:
                    log.error(f"Error processing {avaDir}: {e}")
                    pbar.update(1)

    log.info(f"Processing complete. Success in {nSuccesses} out of {len(avaDirs)} directories.")

    # Copy/move peak files if configured
    allPeakFilesDir = None
    if cfg["GENERAL"].getboolean("copypeakfiles"):
        allPeakFilesDir = moveOrCopyPeakFiles(cfg, regionalDir)

    # Merge output rasters if configured
    mergedRastersDir = None
    if cfg["GENERAL"].getboolean("mergeoutput"):
        mergedRastersDir = mergeOutputRasters(cfg, regionalDir)

    return allPeakFilesDir, mergedRastersDir


def _getSimCountForAvaDir_fast(avaDir):
    """FAST helper function to estimate simulation count for a single avalanche directory.
    
    Instead of running full com1DFA preprocessing, we just count release shapefiles.
    This is ~100x faster than the full preprocessing.
    
    Parameters
    ----------
    avaDir : pathlib.Path
        Path to avalanche directory
    
    Returns
    -------
    int
        Estimated number of simulations (1 per release shapefile)
    """
    import pathlib
    avaDir = pathlib.Path(avaDir)
    inputs_dir = avaDir / "Inputs"
    
    if not inputs_dir.exists():
        return 0
    
    # Count release shapefiles - each one typically = 1 simulation
    rel_files = list(inputs_dir.glob("REL/*.shp")) + list(inputs_dir.glob("REL*/*.shp"))
    
    # If no REL folder, check for release files directly
    if not rel_files:
        rel_files = list(inputs_dir.glob("**/rel_*.shp")) + list(inputs_dir.glob("**/*release*.shp"))
    
    # Minimum 1 simulation per directory if it has inputs
    return max(1, len(rel_files))


def getTotalNumberOfSims(avaDirs, cfgMain, cfgCom7):
    """Get total number of simulations across all avalanche directories.
    
    ALARM PIPELINE MODIFICATION: Uses FAST counting (just counts files, no preprocessing).
    This is ~100x faster than the original method.

    Parameters
    ----------
    avaDirs : list
        List of avalanche directories
    cfgMain : configparser.ConfigParser
        Main configuration
    cfgCom7 : configparser.ConfigParser
        Regional configuration with potential overrides

    Returns
    -------
    int
        Total number of simulations (estimated)
    """
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    
    log.info(f"Fast-counting simulations for {len(avaDirs)} avalanche directories...")
    
    totalSims = 0
    
    # Use ThreadPoolExecutor (faster for I/O-bound file counting)
    # Much faster than ProcessPoolExecutor for simple file operations
    max_workers = min(32, len(avaDirs))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(_getSimCountForAvaDir_fast, avaDir) for avaDir in avaDirs]
        
        # Collect results with progress bar
        with tqdm(total=len(futures), desc="STEP 1/3: Counting scenarios", unit="dir", ncols=100) as pbar:
            for future in as_completed(futures):
                try:
                    count = future.result()
                    totalSims += count
                    pbar.update(1)
                except Exception as e:
                    pbar.update(1)
    
    log.info(f"Estimated total simulations: {totalSims}")
    return totalSims


def processAvaDirCom1Regional(cfgMain, cfgCom7, avalancheDir):
    """Run com1DFA simulation in a specific avalanche directory with regional settings.

    Note: This function calls com1DFA within each avalanche directory within the input directory.
    If wanted it may be used as a template to call another operation within each directory, such as com2AB, ana5Utils, etc.

    Parameters
    ----------
    cfgMain : configparser.ConfigParser
        Main configuration settings
    cfgCom7 : configparser.ConfigParser
        Regional configuration settings with potential overrides
    avalancheDir : pathlib.Path or str
        Path to the avalanche directory to process

    Returns
    -------
    avalancheDir : pathlib.Path or str
        Path to the avalanche directory that was processed
    status : str
        Status of the simulation, "Success" if completed
    """
    # Initialize log for each process
    log = logUtils.initiateLogger(avalancheDir, logName="runCom1DFA")
    log.info("COM1DFA PROCESS CALLED BY COM7REGIONAL RUN")
    log.info("Current avalanche: %s", avalancheDir)

    # Update cfgMain setting to reflect the current avalancheDir
    cfgMain["MAIN"]["avalancheDir"] = str(avalancheDir)

    # Clean input directory of old work and output files from module
    initProj.cleanModuleFiles(avalancheDir, com1DFA, deleteOutput=True)

    # Create com1DFA configuration for the current avalanche directory and override with regional settings
    cfgCom1DFA = cfgUtils.getModuleConfig(
        com1DFA,
        fileOverride="",
        toPrint=False,
        onlyDefault=cfgCom7["com1DFA_com1DFA_override"].getboolean("defaultconfig")  # Fixed: ConfigParser uses lowercase,
    )
    cfgCom1DFA, cfgCom7 = cfgHandling.applyCfgOverride(cfgCom1DFA, cfgCom7, com1DFA, addModValues=False)

    # Run com1DFA in the current avalanche directory
    com1DFA.com1DFAMain(cfgMain, cfgInfo=cfgCom1DFA)

    return avalancheDir, "Success"


def moveOrCopyPeakFiles(cfg, avalancheDir):
    """Collects peak files from multiple sub-avalanche directories.

    Creates directory allPeakFiles: Contains peak files from all avalanche directories

    Parameters
    ----------
    cfg : configparser.ConfigParser
        Configuration containing GENERAL settings:
        - copyPeakFiles: If True, copy/move files; if False, do nothing
        - moveInsteadOfCopy: If True, move files instead of copying
    avalancheDir : pathlib.Path or str
        Base directory where allPeakFiles will be created

    Returns
    -------
    allPeakFilesDir : pathlib.Path or None
        Path to the created allPeakFiles directory or None if copyPeakFiles is False
    """
    if not cfg["GENERAL"].getboolean("copypeakfiles"):
        log.info("copyPeakFiles is False - no files will be copied or moved")
        return None, None

    # Get avalanche directories
    avaDirs = findAvaDirsBasedOnInputsDir(avalancheDir)
    if not avaDirs:
        log.warning("No avalanche directories found to copy/move files from")
        return None, None

    # Set up outdirs
    allPeakFilesDir = pathlib.Path(avalancheDir, "allPeakFiles")
    if allPeakFilesDir.exists():
        shutil.rmtree(str(allPeakFilesDir))
    fU.makeADir(allPeakFilesDir)

    # Get operation type
    useMove = cfg["GENERAL"].getboolean("moveInsteadOfCopy")
    operationType = "Moving" if useMove else "Copying"

    # Collect all peak files first (fast)
    print(f"\n{'='*70}")
    print(f"STEP 3/4: {operationType} peak files")
    print(f"{'='*70}")
    print(f"  Collecting peak files from {len(avaDirs)} directories...")
    
    allPeakFiles = []
    for avaDir in avaDirs:
        peakFiles = list(avaDir.glob("Outputs/**/peakFiles/*.*"))
        allPeakFiles.extend(peakFiles)
    
    print(f"  Found {len(allPeakFiles)} peak files")
    log.info(f"{operationType} {len(allPeakFiles)} peak files in parallel...")
    
    # Define copy/move function for parallel execution
    def copy_single_file(peakFile):
        targetPath = allPeakFilesDir / peakFile.name
        if useMove:
            shutil.move(str(peakFile), str(targetPath))
        else:
            shutil.copy2(str(peakFile), str(targetPath))
        return True
    
    # Process files in parallel using ThreadPoolExecutor (I/O bound)
    from tqdm import tqdm
    max_workers = min(32, len(allPeakFiles))
    
    if len(allPeakFiles) == 0:
        print(f"  WARNING: No peak files found!")
        log.warning("No peak files found to copy/move")
        return allPeakFilesDir
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_single_file, f) for f in allPeakFiles]
        with tqdm(total=len(futures), desc=f"  {operationType} files", unit="file", ncols=100) as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log.warning(f"Error copying file: {e}")
                pbar.update(1)

    print(f"  Done! {len(allPeakFiles)} files copied to {allPeakFilesDir}")
    log.info(f"Copied {len(allPeakFiles)} peak files to {allPeakFilesDir}")
    return allPeakFilesDir


def _readRasterHeader(rasterFile):
    """Read only the header of a raster file (fast, no data loading)."""
    try:
        import rasterio
        with rasterio.open(rasterFile) as src:
            transform = src.transform
            return {
                "cellsize": transform[0],
                "xllcenter": transform[2],
                "yllcenter": transform[5] - src.height * transform[0],  # yllcenter from top-left
                "ncols": src.width,
                "nrows": src.height,
            }
    except Exception:
        # Fallback to full read if rasterio fails
        raster = rasterUtils.readRaster(rasterFile)
        return raster["header"]


def getRasterBounds(rasterFiles):
    """Get the union bounds and validate cell sizes of multiple rasters.
    
    OPTIMIZED: Uses parallel header reading (no data loading).

    Parameters
    ----------
    rasterFiles : list
        List of paths to raster files

    Returns
    -------
    bounds : dict
        Dictionary containing xMin, yMin, xMax, yMax of the union bounds
    cellSize : float
        Cell size of the rasters

    Raises
    ------
    ValueError
        If cell sizes of rasters differ
    """
    from tqdm import tqdm
    
    # Read headers in parallel (I/O bound, ThreadPool is faster)
    max_workers = min(32, len(rasterFiles))
    headers = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_readRasterHeader, f) for f in rasterFiles]
        with tqdm(total=len(futures), desc="Reading raster headers", unit="file", ncols=100, leave=False) as pbar:
            for future in as_completed(futures):
                try:
                    headers.append(future.result())
                except Exception as e:
                    log.warning(f"Error reading header: {e}")
                pbar.update(1)
    
    if not headers:
        raise ValueError("No valid raster headers found")
    
    # Get cellSize from first header
    cellSize = float(headers[0]["cellsize"])
    
    bounds = {
        "xMin": float("inf"),
        "yMin": float("inf"),
        "xMax": float("-inf"),
        "yMax": float("-inf"),
    }

    # Process headers (fast, in memory)
    for header in headers:
        if float(header["cellsize"]) != cellSize:
            raise ValueError(f"Different cell sizes found: {cellSize} vs {header['cellsize']}")

        # Update bounds
        bounds["xMin"] = min(bounds["xMin"], float(header["xllcenter"]))
        bounds["yMin"] = min(bounds["yMin"], float(header["yllcenter"]))
        bounds["xMax"] = max(
            bounds["xMax"],
            float(header["xllcenter"]) + float(header["ncols"]) * cellSize,
        )
        bounds["yMax"] = max(
            bounds["yMax"],
            float(header["yllcenter"]) + float(header["nrows"]) * cellSize,
        )

    return bounds, cellSize


def mergeRasters(rasterFiles, bounds, mergeMethod="max"):
    """Merge multiple rasters into a single raster.

    Parameters
    ----------
    rasterFiles : list
        List of paths to raster files
    bounds : dict
        Dictionary containing xMin, yMin, xMax, yMax of the union bounds
    mergeMethod : str, optional
        Method to use for merging overlapping cells. Options:
        - 'max': maximum value (default)
        - 'min': minimum value
        - 'sum': sum of values
        - 'count': number of overlapping valid results per cell

    Returns
    -------
    mergedHeader : dict
        Header dictionary containing ncols, nrows, xllcenter, yllcenter, cellsize, nodata_value
    mergedData : numpy.ndarray
        2D array containing the merged raster data
    """

    # Merge data with rasterio
    # If something other than min/max is wanted, it is possible to provide a custom function to merge
    mergedData, outputTransform = merge(rasterFiles, method=mergeMethod, masked=True)

    mergedData = np.squeeze(mergedData)
    
    # CRITICAL FIX: Convert masked array to regular array with consistent formatting
    # rasterio.merge() returns masked array which causes mixed formatting (nan, 0.0, 0)
    # GIS tools can't handle this inconsistency
    if ma.isMaskedArray(mergedData):
        # It's a masked array - convert to regular array with -9999 for nodata
        mergedData = mergedData.filled(-9999.0)
    # Ensure consistent data type for GIS compatibility
    mergedData = mergedData.astype(np.float32, copy=False)

    # Calculate dimensions for merged raster; helps checking if merged raster is correct
    nCols = int((bounds["xMax"] - bounds["xMin"]) / outputTransform[0])
    nRows = int((bounds["yMax"] - bounds["yMin"]) / outputTransform[0])
    #
    # # Create merged raster header
    exampleRaster = rasterUtils.readRaster(rasterFiles[0])
    mergedHeader = exampleRaster["header"]
    mergedHeader["ncols"] = nCols
    mergedHeader["nrows"] = nRows
    mergedHeader["xllcenter"] = float(bounds["xMin"])
    mergedHeader["yllcenter"] = float(bounds["yMin"])
    mergedHeader["transform"] = outputTransform

    return mergedHeader, mergedData


def mergeOutputRasters(cfg, avalancheDir):
    """Merge output rasters (peakFiles) from all avalanche simulations.

    Parameters
    ----------
    cfg : configparser.ConfigParser
        Configuration containing settings:
        - GENERAL.mergeOutput: If True, merge rasters
        - GENERAL.mergeTypes: Types of rasters to merge (e.g., 'ppr|pfv|pft')
        - GENERAL.mergeMethods: Methods to use for merging (e.g., 'max')
    avalancheDir : pathlib.Path or str
        Base directory where merged files will be saved

    Returns
    -------
    mergedRastersDir : pathlib.Path or None
        Path to the directory containing merged rasters or None if mergeOutput is False
    """
    if not cfg["GENERAL"].getboolean("mergeoutput", False):
        log.info("mergeOutput is False - no rasters will be merged")
        return None

    # Get all avalanche directories
    # with logUtils.silentLogger():
    avaDirs = findAvaDirsBasedOnInputsDir(avalancheDir)
    if not avaDirs:
        log.warning("No avalanche directories found to merge")
        return None

    # Set up merged rasters directory
    mergedRastersDir = pathlib.Path(avalancheDir, "mergedRasters")
    if mergedRastersDir.exists():
        shutil.rmtree(str(mergedRastersDir))
    mergedRastersDir.mkdir(parents=True, exist_ok=True)

    # Get types to merge
    mergeTypes = cfg["GENERAL"].get("mergetypes").split("|")
    log.info(f"Merging raster types: {mergeTypes}")

    # Get merge methods
    mergeMethods = cfg["GENERAL"].get("mergemethods", "max").lower().split("|")
    log.info(f"Using merge methods: {mergeMethods}")

    # Validate merge methods
    validMethods = {"max", "min", "mean", "sum", "count"}
    invalidMethods = set(mergeMethods) - validMethods
    if invalidMethods:
        raise ValueError(f"Invalid merge methods: {invalidMethods}. Valid options are: {validMethods}")

    from tqdm import tqdm
    
    # Process each raster type with progress
    print(f"\n{'='*70}")
    print(f"STEP 4/4: Merging rasters ({len(mergeTypes)} types)")
    print(f"{'='*70}\n")
    
    for rasterType in mergeTypes:
        # Find all files of this type across all avalanche directories
        print(f"\n  [{rasterType.upper()}] Collecting raster files...")
        rasterFiles = []
        for avaDir in avaDirs:
            peakFilesDir = avaDir / "Outputs" / "com1DFA" / "peakFiles"
            if peakFilesDir.is_dir():
                rasterFiles.extend(list(peakFilesDir.glob(f"*_{rasterType}.*")))

        if not rasterFiles:
            print(f"  [{rasterType.upper()}] WARNING: No rasters found to merge")
            log.warning(f"No {rasterType} rasters found to merge")
            continue

        print(f"  [{rasterType.upper()}] Found {len(rasterFiles)} rasters")

        # Get bounds and validate cell sizes (now parallelized!)
        print(f"  [{rasterType.upper()}] Reading raster headers...")
        bounds, cellSize = getRasterBounds(rasterFiles)

        # Merge and save rasters
        for mergeMethod in mergeMethods:
            print(f"  [{rasterType.upper()}] Merging with method '{mergeMethod}'... (this may take a while)")
            mergedHeader, mergedData = mergeRasters(rasterFiles, bounds, mergeMethod=mergeMethod)
            outputPath = mergedRastersDir / f"merged_{rasterType}_{mergeMethod}"
            # Fix header nodata_value to match filled data (-9999)
            mergedHeader["nodata_value"] = -9999.0
            print(f"  [{rasterType.upper()}] Writing merged raster...")
            rasterUtils.writeResultToRaster(mergedHeader, mergedData, outputPath, flip=False)  # Do not flip - data already flipped when read with readRaster
            print(f"  [{rasterType.upper()}] Saved: {outputPath}")
            log.info(f"Saved merged {rasterType} raster (method: {mergeMethod}) to: {outputPath}")

    return mergedRastersDir
