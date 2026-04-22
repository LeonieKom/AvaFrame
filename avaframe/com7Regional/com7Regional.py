"""Module for handling regional avalanche simulations."""

import os
import math
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

import rasterio
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

    # -------------------------------------------------------------------------
    # Dynamic parallelism: decide how many PRA simulations run in parallel.
    #
    # Parallelization levels (outer → inner):
    #   L1  regional_runner  → N cells in parallel   (ThreadPoolExecutor)
    #   L2  com7Regional     → M PRAs in parallel     (ProcessPoolExecutor)  ← HERE
    #   L3  com1DFA per PRA  → 1 process              (always 1, set below)
    #   L4  Numba threads    → T threads per process   (NUMBA_NUM_THREADS)
    #
    # Goal: L1 × L2 ≈ total_cores  so the machine is busy but not overloaded.
    #
    # ALARM_MAX_WORKERS_PER_CELL (set by regional_runner) is the *default*
    # worker budget for this cell.  It is calculated as cores / max_cells and
    # works well for small cells.  For cells with many PRAs we can safely use
    # all of this budget; for cells with very few PRAs we cap at nAvaDirs so
    # we don't spawn more workers than there is work.
    # -------------------------------------------------------------------------
    import multiprocessing

    nAvaDirs = len(avaDirs)
    totalCores = multiprocessing.cpu_count() or 4

    # Read the per-cell worker budget from the environment (set by regional_runner)
    envWorkers = os.environ.get('ALARM_MAX_WORKERS_PER_CELL')
    if envWorkers:
        workerBudget = int(envWorkers)
    else:
        # Standalone / non-regional mode: use cfgMain nCPU as before
        workerBudget = cfgUtils.getNumberOfProcesses(cfgMain, nAvaDirs)

    # Never use more workers than avalanche directories
    nProcesses = min(workerBudget, nAvaDirs)
    # Sanity: at least 1
    nProcesses = max(1, nProcesses)

    log.info(f"Dynamic parallelism: {nAvaDirs} PRAs, worker budget {workerBudget}, "
             f"using {nProcesses} parallel processes (total cores: {totalCores})")

    # Set nCPU for com1DFA *inside each PRA* to 1.
    # Each PRA runs a single com1DFA simulation — nested parallelism at L3
    # would cause L1 × L2 × L3 processes and overload the machine.
    cfgMain["MAIN"]["nCPU"] = "1"

    # Track progress and results
    completed = 0
    nSuccesses = 0

    # Import tqdm for progress bar
    from tqdm import tqdm
    
    print(f"\n{'='*70}")
    print(f"STEP 2/3: Running {nAvaDirs} avalanche simulations")
    print(f"Using {nProcesses} parallel processes (budget: {workerBudget}, cores: {totalCores})")
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
    try:
        com1DFA.com1DFAMain(cfgMain, cfgInfo=cfgCom1DFA)
        return avalancheDir, "Success"
    finally:
        # Close all handlers to release file locks (critical for Windows)
        for handler in log.handlers[:]:
            handler.close()
            log.removeHandler(handler)


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


def getRasterBounds(rasterFiles, maxExtentKm=100.0):
    """Get the union bounds and validate cell sizes of multiple rasters.
    
    OPTIMIZED: Uses parallel header reading (no data loading).
    Includes outlier detection to prevent OOM from stale/corrupted rasters.

    Parameters
    ----------
    rasterFiles : list
        List of paths to raster files
    maxExtentKm : float, optional
        Maximum allowed extent in km for either axis (default: 100 km).
        If exceeded after outlier filtering, raises ValueError.

    Returns
    -------
    bounds : dict
        Dictionary containing xMin, yMin, xMax, yMax of the union bounds
    cellSize : float
        Cell size of the rasters

    Raises
    ------
    ValueError
        If cell sizes of rasters differ or merged extent is unreasonably large
    """
    from tqdm import tqdm
    
    # Read headers in parallel (I/O bound, ThreadPool is faster)
    max_workers = min(32, len(rasterFiles))
    headers = []
    header_files = []  # track which file each header belongs to
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(_readRasterHeader, f): f for f in rasterFiles}
        with tqdm(total=len(future_to_file), desc="Reading raster headers", unit="file", ncols=100, leave=False) as pbar:
            for future in as_completed(future_to_file):
                try:
                    hdr = future.result()
                    headers.append(hdr)
                    header_files.append(future_to_file[future])
                except Exception as e:
                    log.warning(f"Error reading header: {e}")
                pbar.update(1)
    
    if not headers:
        raise ValueError("No valid raster headers found")
    
    # Get cellSize from first header
    cellSize = float(headers[0]["cellsize"])
    
    # --- Outlier detection (median-based) ---
    # Compute centre coordinate for each raster; flag any that are far from
    # the median as potential stale/corrupted files from a previous run.
    if len(headers) >= 3:
        centres_x = []
        centres_y = []
        for h in headers:
            cx = float(h["xllcenter"]) + float(h["ncols"]) * cellSize / 2.0
            cy = float(h["yllcenter"]) + float(h["nrows"]) * cellSize / 2.0
            centres_x.append(cx)
            centres_y.append(cy)
        
        med_x = float(np.median(centres_x))
        med_y = float(np.median(centres_y))
        
        # Threshold: any raster whose centre is > maxExtentKm/2 from the median
        # is almost certainly from a different location (e.g. stale run)
        threshold_m = maxExtentKm * 1000.0 / 2.0
        
        keep_idx = []
        drop_idx = []
        for i, (cx, cy) in enumerate(zip(centres_x, centres_y)):
            dist = ((cx - med_x)**2 + (cy - med_y)**2) ** 0.5
            if dist > threshold_m:
                drop_idx.append(i)
            else:
                keep_idx.append(i)
        
        if drop_idx:
            log.warning(f"Outlier detection: {len(drop_idx)} of {len(headers)} rasters are "
                        f"far from median centre ({med_x:.0f}, {med_y:.0f}). Dropping them:")
            for i in drop_idx:
                log.warning(f"  DROPPED: {header_files[i]} "
                            f"(centre: {centres_x[i]:.0f}, {centres_y[i]:.0f}, "
                            f"dist: {((centres_x[i]-med_x)**2+(centres_y[i]-med_y)**2)**0.5/1000:.1f} km)")
            
            headers = [headers[i] for i in keep_idx]
            header_files = [header_files[i] for i in keep_idx]
            # Also filter the rasterFiles list so the caller can use it
            rasterFiles[:] = [rasterFiles[i] for i in keep_idx]
            
            if not headers:
                raise ValueError("All rasters were filtered as outliers — no valid rasters remain")
    
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

    # --- Final extent sanity check ---
    extent_x_km = (bounds["xMax"] - bounds["xMin"]) / 1000.0
    extent_y_km = (bounds["yMax"] - bounds["yMin"]) / 1000.0
    log.info(f"Merged extent: {extent_x_km:.1f} km x {extent_y_km:.1f} km "
             f"({len(headers)} rasters, cellSize={cellSize} m)")
    
    if extent_x_km > maxExtentKm or extent_y_km > maxExtentKm:
        n_pixels_x = int((bounds["xMax"] - bounds["xMin"]) / cellSize)
        n_pixels_y = int((bounds["yMax"] - bounds["yMin"]) / cellSize)
        mem_gb = n_pixels_x * n_pixels_y * 4 / (1024**3)  # float32
        raise ValueError(
            f"Merged extent ({extent_x_km:.0f} km x {extent_y_km:.0f} km) exceeds "
            f"maximum allowed ({maxExtentKm} km). This would require ~{mem_gb:.1f} GiB. "
            f"Likely cause: stale output files from a previous run with different DEM. "
            f"Delete the cell's 3_SplitInputs directory and re-run."
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

    # Check if input rasters are uint16 compressed (scaled PPR values)
    # We need to read scale_factor from metadata to de-scale after merging
    scale_factor = None
    with rasterio.open(rasterFiles[0]) as src:
        if src.dtypes[0] == 'uint16':
            tags = src.tags()
            scale_factor = float(tags.get('scale_factor', 10.0))
            log.info(f"Detected uint16 compressed rasters with scale_factor={scale_factor}")
    
    # Merge data with rasterio
    # If something other than min/max is wanted, it is possible to provide a custom function to merge
    mergedData, outputTransform = merge(rasterFiles, method=mergeMethod, masked=True)

    mergedData = np.squeeze(mergedData)
    
    # CRITICAL FIX: Convert masked array to regular array with consistent formatting
    # rasterio.merge() returns masked array which causes mixed formatting (nan, 0.0, 0)
    # GIS tools can't handle this inconsistency
    # 
    # IMPORTANT: If input rasters are uint16 (compressed PPR), we must convert to float32
    # BEFORE calling filled(), because -9999 doesn't fit in uint16 (range 0-65535)
    if ma.isMaskedArray(mergedData):
        # First convert to float32 to allow -9999 nodata value
        mergedData = mergedData.astype(np.float32)
        # Then fill masked values with -9999
        mergedData = mergedData.filled(-9999.0)
    else:
        # Ensure consistent data type for GIS compatibility
        mergedData = mergedData.astype(np.float32, copy=False)
    
    # De-scale uint16 compressed data back to original values (e.g., kPa)
    if scale_factor is not None:
        # Only de-scale valid data (not nodata values)
        valid_mask = mergedData != -9999.0
        mergedData[valid_mask] = mergedData[valid_mask] / scale_factor
        log.info(f"De-scaled merged data by factor {scale_factor}")

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


def _is_simulation_dem_clipped(peakFile, min_run_px=20):
    """Check whether a simulation was clipped at the DEM boundary by inspecting
    its peak-file raster.

    **Physical principle**: when a simulation is clipped at the DEM edge, the
    avalanche has non-zero pressure right up to the last row or column of the
    raster — it would have continued beyond the domain but was artificially
    stopped.  The result is a straight-line artefact in the merged raster.

    By contrast, a simulation that naturally stops inside the DEM (valley floor,
    loss of momentum) has *no* valid data pixels at the raster border.  This is
    because ``com1DFA`` writes PPR rasters with a bounding box that matches the
    last non-zero pixel, not the full DEM extent.  A "natural stop" may have a
    bounding box that touches the DEM boundary (the raster origin aligns with a
    DEM grid corner), but its last pixel row/column will be zero.

    The test: find the longest run of consecutive valid-data pixels (value > 0
    and != nodata) in any of the four border rows/columns.  If this run is at
    least ``min_run_px`` pixels long, the simulation is flagged as clipped.
    The minimum-run threshold avoids false positives from isolated pixels that
    touch the edge due to numerical noise.

    Parameters
    ----------
    peakFile : pathlib.Path or str
        Path to the individual peak-file raster (e.g. ``individual_8_..._ppr.tif``).
    min_run_px : int, optional
        Minimum consecutive valid-pixel run length at the raster border required
        to flag the simulation as clipped.  Default is 20 (= 100 m at 5 m
        resolution).  Short runs (< 100 m) are unlikely to produce visible
        straight-line artefacts in the merged output.

    Returns
    -------
    bool
        ``True`` if the simulation appears to be DEM-clipped, ``False`` otherwise.
        Returns ``False`` on any read error so that uncertain files are kept.
    """
    try:
        with rasterio.open(peakFile) as src:
            data = src.read(1)
            nodata = src.nodata
    except Exception as exc:
        log.debug("_is_simulation_dem_clipped: could not read %s: %s", peakFile, exc)
        return False

    if nodata is not None:
        valid = (data > 0) & (data != nodata)
    else:
        valid = data > 0

    def _max_run(arr):
        """Longest consecutive True run in a 1-D boolean array."""
        run = max_run = 0
        for v in arr:
            if v:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        return max_run

    # Check the single outermost row/column on all four sides
    return (
        _max_run(valid[0, :])   >= min_run_px or   # north border
        _max_run(valid[-1, :])  >= min_run_px or   # south border
        _max_run(valid[:, 0])   >= min_run_px or   # west border
        _max_run(valid[:, -1])  >= min_run_px       # east border
    )


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
    
    # Pre-compute which simulations are DEM-clipped (once, shared across all rasterTypes).
    #
    # Detection: a simulation whose PPR raster has consecutive valid-data pixels
    # (pressure > 0) reaching the very last row or column of the raster was
    # artificially stopped at the DEM boundary.  The avalanche would have
    # continued beyond the domain — this produces a straight-line artefact in the
    # merged output.  By contrast, a natural stop inside the DEM leaves the border
    # row/column empty (com1DFA crops to the data bounding box, so the last pixel
    # of a natural stop carries no pressure).
    #
    # We use PPR as the reference type because it is always produced.  The same
    # avaDir is then excluded for ALL raster types (ppr, pfv, pft) for consistency.
    clipped_avaDirs = set()
    reference_type = "ppr"

    # Step 1a: fast flag-file check — com1DFA writes a *_BOUNDARY_EXIT.flag
    # file to the peakFiles directory whenever nExitedParticles > 0.  This is
    # the authoritative indicator that particles left the domain and that the
    # PPR raster carries artificial boundary-accumulation pressure.
    n_flag = 0
    for avaDir in avaDirs:
        peakFilesDir = avaDir / "Outputs" / "com1DFA" / "peakFiles"
        if not peakFilesDir.is_dir():
            continue
        if list(peakFilesDir.glob("*_BOUNDARY_EXIT.flag")):
            clipped_avaDirs.add(avaDir)
            n_flag += 1

    if n_flag:
        log.warning(
            "BOUNDARY EXIT flags: %d simulation directories had particles exit the DEM. "
            "These will be excluded from the merge.",
            n_flag,
        )
        print(f"\n  [BOUNDARY FILTER] {n_flag} simulations excluded via _BOUNDARY_EXIT.flag")

    # Step 1b: pixel-based fallback check for simulations that ran before the
    # flag-file mechanism was introduced (legacy results without flag files).
    # Flag each avaDir whose PPR raster has valid-data pixels at the
    # raster border — a minimum run of 20 pixels (100 m at 5 m) is required.
    for avaDir in avaDirs:
        if avaDir in clipped_avaDirs:
            continue  # already flagged above
        peakFilesDir = avaDir / "Outputs" / "com1DFA" / "peakFiles"
        if not peakFilesDir.is_dir():
            continue
        ref_files = list(peakFilesDir.glob(f"*_{reference_type}.*"))
        for ref_file in ref_files:
            if _is_simulation_dem_clipped(ref_file, min_run_px=20):
                clipped_avaDirs.add(avaDir)
                break

    if clipped_avaDirs:
        log.warning(
            "DEM-clip filter: excluding %d of %d simulation directories whose avalanche "
            "reached the DEM boundary (straight-line artefact suppression).",
            len(clipped_avaDirs), len(avaDirs)
        )
        print(f"\n  [CLIP FILTER] Excluding {len(clipped_avaDirs)}/{len(avaDirs)} "
              f"DEM-clipped simulations from merge.")
    else:
        log.info("DEM-clip filter: no clipped simulations detected.")

    for rasterType in mergeTypes:
        # Find all files of this type across all avalanche directories
        print(f"\n  [{rasterType.upper()}] Collecting raster files...")
        rasterFiles = []
        n_excluded = 0
        for avaDir in avaDirs:
            if avaDir in clipped_avaDirs:
                n_excluded += 1
                continue
            peakFilesDir = avaDir / "Outputs" / "com1DFA" / "peakFiles"
            if peakFilesDir.is_dir():
                rasterFiles.extend(list(peakFilesDir.glob(f"*_{rasterType}.*")))

        if n_excluded:
            print(f"  [{rasterType.upper()}] Excluded {n_excluded} DEM-clipped simulations.")

        if not rasterFiles:
            print(f"  [{rasterType.upper()}] WARNING: No rasters found to merge")
            log.warning(f"No {rasterType} rasters found to merge")
            continue

        print(f"  [{rasterType.upper()}] Found {len(rasterFiles)} rasters ({n_excluded} excluded as DEM-clipped)")

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

            # Strip the outermost 50 m (10 pixels at 5 m resolution) from the
            # merged raster on all four sides.  com1DFA deposits particle mass
            # in a thin strip at the DEM boundary when particles lose SPH
            # neighbours on the outer side, producing a characteristic
            # straight-line artifact in the merged output.  The strip is always
            # within the overlap zone between neighbouring tiles (≥ 2 km), so
            # the regional MAX merge will fill the masked pixels with correct
            # values from the neighbour tile.  This also ensures that the
            # avalanche track exporter (which uses this cell-level PPR) does
            # not inherit the artifact geometry.
            _strip_margin_m = 50.0
            _csz = mergedHeader.get("cellsize", 5.0)
            _margin_px = max(1, int(math.ceil(_strip_margin_m / _csz)))
            _nd = mergedHeader.get("nodata_value", -9999.0)
            if mergedData.ndim == 2:
                _nr, _nc = mergedData.shape
                if _nr > 2 * _margin_px and _nc > 2 * _margin_px:
                    mergedData[:_margin_px, :]  = _nd
                    mergedData[-_margin_px:, :] = _nd
                    mergedData[:, :_margin_px]  = _nd
                    mergedData[:, -_margin_px:] = _nd
            print(f"  [{rasterType.upper()}] Writing merged raster...")
            
            # Apply uint16 compression for PPR if output is GeoTIFF
            compress_uint16 = (rasterType == "ppr" and mergedHeader.get("driver") == "GTiff")
            scale_factor = 10.0 if compress_uint16 else None
            
            rasterUtils.writeResultToRaster(mergedHeader, mergedData, outputPath, flip=False,
                                           compress_uint16=compress_uint16, scale_factor=scale_factor)
            print(f"  [{rasterType.upper()}] Saved: {outputPath}")
            log.info(f"Saved merged {rasterType} raster (method: {mergeMethod}) to: {outputPath}")


    return mergedRastersDir
