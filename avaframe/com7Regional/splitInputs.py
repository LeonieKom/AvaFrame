"""Module for splitting and organizing regional avalanche input data."""

import logging
import shapefile  # pyshp
from shapely.geometry import box
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import os
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from avaframe.in2Trans import rasterUtils
from avaframe.in3Utils import fileHandlerUtils as fU
from avaframe.in1Data import getInput
from avaframe.in3Utils.initializeProject import initializeFolderStruct
from avaframe.in2Trans import shpConversion as shpConv
from avaframe.out3Plot.outCom7Regional import createReportPlot

# create local logger
log = logging.getLogger(__name__)


def splitInputsMain(avalancheDir, outputDir, cfg, cfgMain):
    """Process and organize avalanche input data into individual avalanche directories based
    on release area's "group" and "scenario" attributes provided in the release area file. If no
    "group" attribute is provided, one avalanche directory per feature will be created (scenario is
    ignored in this case).

    Parameters
    ----------
    avalancheDir : pathlib.Path object
        Path to input directory containing release areas (REL) and DEM files
    outputDir : pathlib.Path object
        Path to output directory where organized folders will be created
    cfg : dict
        Configuration settings containing:
        - GENERAL.bufferSize : float, buffer size for DEM clipping
    cfgMain : dict
        Configuration settings containing:
        - FLAGS.createReport : bool, whether to write report
        - FLAGS.savePlot : bool, whether to save plots

    Returns
    -------
    none

    Notes
    -----
    Expected input directory structure:
    avalancheDir/
    └── Inputs/
        ├── REL/
        │   └── *.shp         # all release areas
        ├── ENT/              # all entrainment areas (optional)
        │   └── *.shp
        ├── RES/              # all resistance areas (optional)
        │   └── *.shp
        └── *.asc or *.tif    # digital elevation model (DEM)
    """
    # Fetch the necessary input
    inputSimFilesAll = getInput.getInputDataCom1DFA(avalancheDir)

    # extract release shapefile, make sure only one exists
    if len(inputSimFilesAll["relFiles"]) == 1:
        inputShp = inputSimFilesAll["relFiles"][0]
    else:
        log.error(f"Expected only one release area file, found {len(inputSimFilesAll['relFiles'])}.")
        return

    # Get the input DEM
    inputDEM = getInput.getDEMPath(avalancheDir)

    # Create the output directory
    fU.makeADir(outputDir)

    # Step 1: Create the directory list
    log.info("Initializing folder structure for each group...")
    dirListGrouped = createDirList(inputShp)
    log.info("Finished creating folder list")

    # Step 2: Set up avalanche directories
    log.info("Initializing folder structure for each entry...")
    n_groups = len(dirListGrouped)
    print(f"Creating {n_groups} avalanche directories...")
    
    # Batch directory creation for better I/O performance
    # Instead of calling initializeFolderStruct per directory (slow on network storage),
    # create all directories in one pass
    t_start = time.time()
    
    # Collect all directory paths to create
    all_dirs_to_create = []
    for entry in dirListGrouped:
        dirName = entry["dirName"]
        base_dir = outputDir / dirName
        # Standard AvaFrame structure
        subdirs = [
            base_dir / "Inputs" / "REL",
            base_dir / "Inputs" / "ENT", 
            base_dir / "Inputs" / "RES",
            base_dir / "Outputs" / "com1DFA" / "peakFiles",
            base_dir / "Outputs" / "com1DFA" / "particles",
        ]
        all_dirs_to_create.extend(subdirs)
    
    # Create all directories in batch
    for dir_path in all_dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Created {n_groups} directory structures in {time.time()-t_start:.1f}s")
    log.info("Finished folder initialization")

    # Step 3: Split and move release areas to each directory
    log.info("Splitting and moving release areas...")
    splitAndMoveReleaseAreas(dirListGrouped, inputShp, outputDir)
    log.info("Finished splitting and moving release areas")

    # Step 4: Clip and move DEM
    log.info("Clipping and moving DEM...")
    groupExtents = clipDEMByReleaseGroup(dirListGrouped, inputDEM, outputDir, cfg)
    log.info("Finished clipping and moving of DEM")

    # Step 5: Clip and move optional input (currently only ENT and RES)
    log.info("Clipping and moving optional input...")
    groupFeatures = clipAndMoveOptionalInput(inputSimFilesAll, outputDir, groupExtents)
    log.info("Finished clipping and moving optional input")

    # Step 6: Divide release areas into scenarios
    log.info("Separating release areas by scenarios...")
    splitByScenarios(dirListGrouped, outputDir)
    log.info("Finished separating by scenarios")

    # Step 7: Write reports
    if cfgMain["FLAGS"].getboolean("createReport"):
        log.info("Writing reports...")
        writeScenarioReport(dirListGrouped, outputDir)
        if cfgMain["FLAGS"].getboolean("savePlot"):
            createReportPlots(dirListGrouped, inputDEM, outputDir, groupExtents, groupFeatures)
        log.info("Finished writing reports")


def createDirList(inputShp):
    """Create a list of entries from each feature in the input shapefile, grouped by the 'group' attribute.

    Parameters
    ----------
    inputShp: pathlib.Path object
        path to input shapefile

    Returns
    -------
    dirListGrouped: list
        list of dictionaries containing dirName (group name), properties list, and geometries list,
        where features are grouped by their 'group' attribute
    """
    fields, fieldNames, properties, geometries, srs = shpConv.readShapefile(inputShp)

    # Create dictionary to store groups
    groups = {}
    unnamedCount = 1

    for props, geom in zip(properties, geometries):
        propsLower = {key.lower(): value for key, value in props.items()}  # Handle case sensitivity

        # Get group name from 'group' attribute, fallback to unnamed if not present
        groupName = propsLower.get("group", "").strip() or f"{str(unnamedCount).zfill(5)}"
        if not propsLower.get("group", "").strip():
            unnamedCount += 1
            log.info(f"No 'group' field or empty group found in {inputShp}, using '{groupName}'")

        # Initialize group if not exists
        if groupName not in groups:
            groups[groupName] = {
                "dirName": groupName,
                "properties": [],
                "geometries": [],
            }

        # Add feature to group
        groups[groupName]["properties"].append(props)
        groups[groupName]["geometries"].append(geom)

    # Convert dictionary to list and sort by dirName
    dirListGrouped = list(groups.values())
    dirListGrouped.sort(key=lambda x: x["dirName"].lower())

    # Log total number of features
    totalFeatures = sum(len(group["geometries"]) for group in dirListGrouped)
    log.info(f"Found '{totalFeatures}' features that were organized into '{len(dirListGrouped)}' groups")

    return dirListGrouped


def splitAndMoveReleaseAreas(dirList, inputShp, outputDir):
    """Split release areas into individual shapefiles and write them to their respective folders.

    Parameters
    ----------
    dirList: list
        list of dictionaries containing dirName, properties list, and geometries list
    inputShp: pathlib.Path object
        path to input shapefile
    outputDir: pathlib.Path object
        path to output directory where folders will be created

    Returns
    -------
    none
    """
    # Read the input shapefile
    fields, fieldNames, properties, geometries, srs = shpConv.readShapefile(inputShp)

    featuresByName = {}
    for entry in dirList:
        name = entry["dirName"]  # Get release area name
        # Group entries with the same name
        if name not in featuresByName:
            featuresByName[name] = []
        # add corresponding properties and geometries
        for i, properties in enumerate(entry["properties"]):
            featuresByName[name].append((properties, entry["geometries"][i]))

    # Write shapefiles to their respective folders
    # Use batch writing with progress bar for better I/O performance
    print(f"Writing {len(featuresByName)} release area shapefiles...")
    t_start = time.time()
    
    iterator = tqdm(featuresByName.items(), desc="Writing REL shapefiles", unit="file") if HAS_TQDM else featuresByName.items()
    for name, features in iterator:
        shpOutPath = outputDir / name / "Inputs" / "REL" / name
        shpConv.writeShapefile(shpOutPath, fields, fieldNames, features, srs)
        log.debug(f"Saved release area to '{shpOutPath}'.")
    
    log.info(f"Wrote {len(featuresByName)} REL shapefiles in {time.time()-t_start:.1f}s")


def checkFeatureIsolation(geometries, properties, bufferSize, groupName):
    """Check if any feature in the group is isolated from all others.

    A feature is considered isolated if its buffered bounding box does not overlap
    with any other feature's buffered bounding box in the group.

    Parameters
    ----------
    geometries: list
        List of geometry objects to check
    properties: list
        List of dictionaries containing properties for each geometry
    bufferSize: float
        Buffer size to use when creating bounding boxes
    groupName: str
        Name of the group, used for error messages

    Raises
    ------
    ValueError
        If any feature is isolated from all others in the group
    """
    # Skip check if only one feature
    if len(geometries) <= 1:
        log.debug(f"Group '{groupName}' has only one feature, proceeding without isolation check.")
        return

    # Create buffered bounding boxes for each feature
    boundingBoxes = []
    for geom in geometries:
        center = geom.centroid

        # Calculate bounding box for this feature
        currXMin = center.x - bufferSize
        currYMin = center.y - bufferSize
        currXMax = center.x + bufferSize
        currYMax = center.y + bufferSize

        # Update group extent
        boundingBoxes.append(box(currXMin, currYMin, currXMax, currYMax))

    # Check each feature's bounding box against all others
    for i, bbox in enumerate(boundingBoxes):
        hasOverlap = False
        for j, otherBbox in enumerate(boundingBoxes):
            if i != j and bbox.intersects(otherBbox):
                hasOverlap = True
                break

        if not hasOverlap:
            # Find feature name regardless of case (NAME, name, Name etc.)
            featureProps = {key.lower(): value for key, value in properties[i].items()}
            featureName = featureProps.get("name", f"unnamed feature {i+1}").strip()

            message = f"Feature '{featureName}' in group '{groupName}' is isolated from all other features - consider assigning it to a different group"
            log.error(message)
            raise ValueError(message)


def _compute_flow_direction(raster, header, xMins, yMins, xMaxs, yMaxs):
    """Compute the mean downhill direction from the DEM at a PRA group location.

    Extracts a small DEM patch covering the PRA extent (plus 50 m padding),
    computes the spatial gradient, and returns a normalized downhill vector
    in geographic coordinates.

    Parameters
    ----------
    raster : numpy.ndarray
        Full DEM raster (row 0 = north).
    header : dict
        Raster header with cellsize, xllcenter, yllcenter, nrows, ncols.
    xMins, yMins, xMaxs, yMaxs : list of float
        Per-geometry bounding box coordinates of the PRA group.

    Returns
    -------
    downhill_x : float
        Normalized x-component of the downhill direction (+east, -west).
        0.0 if terrain is flat.
    downhill_y : float
        Normalized y-component of the downhill direction (+north, -south).
        0.0 if terrain is flat.
    slope_deg : float
        Mean slope angle in degrees.
    """
    cellSize = header["cellsize"]
    xOrigin = header["xllcenter"]
    yOrigin = header["yllcenter"]
    nRows = header["nrows"]
    nCols = header["ncols"]

    # PRA extent with small padding (50 m) to get representative gradient
    padding = max(50.0, cellSize * 3)
    xMin = min(xMins) - padding
    xMax = max(xMaxs) + padding
    yMin = min(yMins) - padding
    yMax = max(yMaxs) + padding

    # Convert geographic extent to raster indices (row 0 = top/north)
    colStart = max(0, int((xMin - xOrigin) / cellSize))
    colEnd = min(nCols, int((xMax - xOrigin) / cellSize) + 1)
    # Geographic row indices (0 = south)
    geoRowStart = max(0, int((yOrigin + nRows * cellSize - yMax) / cellSize))
    geoRowEnd = min(nRows, int((yOrigin + nRows * cellSize - yMin) / cellSize))
    # Flip to numpy row indices (0 = north)
    rowStart = nRows - geoRowEnd
    rowEnd = nRows - geoRowStart

    if rowEnd <= rowStart or colEnd <= colStart:
        return 0.0, 0.0, 0.0

    patch = raster[rowStart:rowEnd, colStart:colEnd]

    if patch.size < 9:  # need at least a 3x3 patch
        return 0.0, 0.0, 0.0

    # Mask nodata values (typically -9999 or very large negatives)
    valid_mask = patch > -9000
    if np.sum(valid_mask) < 9:
        return 0.0, 0.0, 0.0

    # Compute gradient on the patch
    # rasterUtils.readRaster applies np.flipud(), so row 0 = SOUTH, row index increases NORTHWARD.
    # axis=0: row gradient — positive = value increases with row index = increases NORTHWARD
    # axis=1: col gradient — positive = value increases with col index = increases EASTWARD
    dy_raster, dx_raster = np.gradient(patch, cellSize)

    # Convert to geographic gradient (positive = uphill direction)
    # geo dz/dx: same as raster dx (positive = elevation increases east)
    # geo dz/dy: same as raster dy (positive = elevation increases north, because row 0 = south)
    geo_dzdx = np.nanmean(np.where(valid_mask, dx_raster, np.nan))
    geo_dzdy = np.nanmean(np.where(valid_mask, dy_raster, np.nan))

    if np.isnan(geo_dzdx) or np.isnan(geo_dzdy):
        return 0.0, 0.0, 0.0

    magnitude = np.sqrt(geo_dzdx**2 + geo_dzdy**2)
    slope_deg = np.degrees(np.arctan(magnitude))

    if magnitude < 1e-10:
        return 0.0, 0.0, slope_deg

    # Downhill direction = opposite of gradient (gradient points uphill)
    downhill_x = -geo_dzdx / magnitude
    downhill_y = -geo_dzdy / magnitude

    return downhill_x, downhill_y, slope_deg


def _write_clipped_dem(args):
    """Helper function to write a single clipped DEM (for parallel execution).
    
    Parameters
    ----------
    args : tuple
        (dirName, clippedData, clippedHeader, outputDir)
    
    Returns
    -------
    tuple
        (dirName, (xMinDEM, xMaxDEM, yMinDEM, yMaxDEM))
    """
    dirName, clippedData, clippedHeader, outputDir = args
    cellSize = clippedHeader["cellsize"]
    
    # Write clipped DEM
    outputDEM = outputDir / dirName / "Inputs" / f"{dirName}_DEM"
    rasterUtils.writeResultToRaster(clippedHeader, clippedData, outputDEM, flip=True)
    
    # Calculate final DEM extents
    xMinDEM = clippedHeader["xllcenter"] + (cellSize * 0.5)
    yMinDEM = clippedHeader["yllcenter"] + (cellSize * 0.5)
    xMaxDEM = clippedHeader["xllcenter"] + (clippedHeader["ncols"] * cellSize) - (cellSize * 0.5)
    yMaxDEM = clippedHeader["yllcenter"] + (clippedHeader["nrows"] * cellSize) - (cellSize * 0.5)
    
    return (dirName, (xMinDEM, xMaxDEM, yMinDEM, yMaxDEM))


def clipDEMByReleaseGroup(dirList, inputDEM, outputDir, cfg, parallel=True, max_workers=None):
    """Clip the DEM to include all features in each release group. Returns an error if any feature in a group is isolated.
    
    Uses parallel I/O for faster processing when many groups need to be clipped.

    Parameters
    ----------
    dirList : list
        List of dictionaries containing dirName, and geometries list
    inputDEM : pathlib.Path
        Path to input DEM file
    outputDir : pathlib.Path
        Path to output directory where clipped DEMs will be saved
    cfg : configparser object
        Configuration settings containing:
            - GENERAL.bufferSize : float
                Size of buffer to add around release areas
    parallel : bool, optional
        If True, use parallel I/O for writing clipped DEMs. Default: True
    max_workers : int, optional
        Maximum number of parallel workers. Default: min(32, os.cpu_count() + 4)

    Returns
    -------
    groupExtents : dict
        Dictionary with dirName as key and (xMin, xMax, yMin, yMax) as value.
        The extents are reduced by one pixel on each side to ensure DEM extents
        are larger than clip extents of other input.
    """
    # Read input DEM once
    demData = rasterUtils.readRaster(inputDEM)
    header = demData["header"]
    raster = demData["rasterData"]
    cellSize = header["cellsize"]
    xOrigin = header["xllcenter"]
    yOrigin = header["yllcenter"]
    nRows = header["nrows"]
    nCols = header["ncols"]

    n_groups = len(dirList)
    bufferSize = cfg["GENERAL"].getfloat("bufferSize")
    
    # Read directional clipping settings
    useDirectional = cfg.has_section("DIRECTIONAL") and cfg["DIRECTIONAL"].getboolean("enabled", fallback=False)
    if useDirectional:
        uphillRatio = cfg["DIRECTIONAL"].getfloat("uphillBufferRatio", fallback=0.25)
        minUphillBuffer = cfg["DIRECTIONAL"].getfloat("minUphillBuffer", fallback=200.0)
        flatThreshold = cfg["DIRECTIONAL"].getfloat("flatTerrainThreshold", fallback=2.0)
        log.info(f"Directional clipping enabled: uphillRatio={uphillRatio}, "
                 f"minUphill={minUphillBuffer}m, flatThreshold={flatThreshold}°")
        dirClipCount = 0  # count how many groups used directional clipping
    
    # Determine number of workers
    # In regional mode, ALARM_MAX_WORKERS_PER_CELL limits per-cell parallelism
    # to prevent I/O contention when multiple cells write clipped DEMs simultaneously
    if max_workers is None:
        env_limit = os.environ.get('ALARM_MAX_WORKERS_PER_CELL')
        if env_limit:
            max_workers = min(int(env_limit), 8)  # Cap at 8 for I/O-bound work
            log.info(f"Regional mode: limiting DEM clip workers to {max_workers}")
        else:
            max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    # Phase 1: Prepare all clips (compute-bound, fast)
    print(f"Preparing {n_groups} DEM clips...")
    write_tasks = []
    groupExtents = {}
    totalPixelsSaved = 0
    
    iterator = tqdm(dirList, desc="Preparing clips", unit="group") if HAS_TQDM else dirList
    for entry in iterator:
        dirName = entry["dirName"]
        geometries = entry["geometries"]
        properties = entry["properties"]

        if not geometries:
            message = f"No geometries found for {dirName}"
            log.error(message)
            raise ValueError(message)

        # Check if any features in the group are isolated
        checkFeatureIsolation(geometries, properties, bufferSize, dirName)

        # Get extent of all geometries in group
        bounds = [geom.bounds for geom in geometries]
        xMins, yMins, xMaxs, yMaxs = zip(*bounds)

        # Calculate extent with buffer — symmetric or directional
        if useDirectional:
            # Compute mean flow direction from DEM gradient at PRA location
            downhill_x, downhill_y, slope_deg = _compute_flow_direction(
                raster, header, xMins, yMins, xMaxs, yMaxs
            )

            if slope_deg >= flatThreshold and (abs(downhill_x) > 1e-6 or abs(downhill_y) > 1e-6):
                # Asymmetric buffer: full downhill, reduced uphill, proportional on sides
                # Each direction gets a factor between uphillRatio and 1.0
                # based on alignment with the downhill vector
                eastFactor  = uphillRatio + (1.0 - uphillRatio) * max(0.0, downhill_x)
                westFactor  = uphillRatio + (1.0 - uphillRatio) * max(0.0, -downhill_x)
                northFactor = uphillRatio + (1.0 - uphillRatio) * max(0.0, downhill_y)
                southFactor = uphillRatio + (1.0 - uphillRatio) * max(0.0, -downhill_y)

                # Apply minimum uphill buffer
                eastBuf  = max(minUphillBuffer, eastFactor * bufferSize)
                westBuf  = max(minUphillBuffer, westFactor * bufferSize)
                northBuf = max(minUphillBuffer, northFactor * bufferSize)
                southBuf = max(minUphillBuffer, southFactor * bufferSize)

                xMin = min(xMins) - westBuf
                xMax = max(xMaxs) + eastBuf
                yMin = min(yMins) - southBuf
                yMax = max(yMaxs) + northBuf

                dirClipCount += 1
                log.debug(f"{dirName}: directional clip (slope={slope_deg:.1f}°, "
                          f"downhill=[{downhill_x:.2f},{downhill_y:.2f}]) → "
                          f"E={eastBuf:.0f} W={westBuf:.0f} N={northBuf:.0f} S={southBuf:.0f} m")
            else:
                # Flat terrain or no clear direction → symmetric buffer
                xMin = min(xMins) - bufferSize
                xMax = max(xMaxs) + bufferSize
                yMin = min(yMins) - bufferSize
                yMax = max(yMaxs) + bufferSize
                log.debug(f"{dirName}: symmetric buffer (slope={slope_deg:.1f}° < {flatThreshold}°)")
        else:
            # Original symmetric buffer
            xMin = min(xMins) - bufferSize
            xMax = max(xMaxs) + bufferSize
            yMin = min(yMins) - bufferSize
            yMax = max(yMaxs) + bufferSize

        # Convert extent to grid indices
        colStart = max(0, int((xMin - xOrigin) / cellSize))
        colEnd = min(nCols, int((xMax - xOrigin) / cellSize) + 1)
        rowStart = max(0, int((yOrigin + nRows * cellSize - yMax) / cellSize))
        rowEnd = min(nRows, int((yOrigin + nRows * cellSize - yMin) / cellSize))

        # Ensure valid row indices
        if rowEnd <= rowStart:
            log.warning(f"Invalid row indices calculated for {dirName}: start={rowStart}, end={rowEnd}")
            continue

        # Flip row indices for bottom-left origin
        rowStart, rowEnd = nRows - rowEnd, nRows - rowStart

        # Clip the DEM data (numpy slicing is very fast)
        clippedData = raster[rowStart:rowEnd, colStart:colEnd].copy()  # copy to avoid memory issues

        # Handle nodata values: Convert original nodata and 0 values to np.nan
        # This ensures they are properly written as nodata in the output raster
        originalNoData = header.get("nodata_value", -9999)
        if np.isnan(originalNoData):
            originalNoData = -9999  # fallback if somehow nan got stored

        # Mask both original nodata values and 0 values as invalid (set to nan)
        # 0 values at DEM edges are artifacts from clipping beyond original DEM extent
        invalid_mask = (clippedData == originalNoData) | (clippedData == 0)
        if np.any(invalid_mask):
            clippedData = clippedData.astype(np.float32)  # ensure float for nan support
            clippedData[invalid_mask] = np.nan
            log.debug(f"{dirName}: Masked {np.sum(invalid_mask)} invalid pixels (nodata/0) as nan")

        # Create header for clipped DEM
        clippedHeader = header.copy()
        clippedHeader["ncols"] = colEnd - colStart
        clippedHeader["nrows"] = rowEnd - rowStart
        clippedHeader["xllcenter"] = xOrigin + colStart * cellSize
        clippedHeader["yllcenter"] = yOrigin + rowStart * cellSize
        clippedHeader["transform"] = rasterUtils.transformFromASCHeader(clippedHeader)
        clippedHeader["nodata_value"] = np.nan  # np.nan will be converted to -9999 by writeResultToRaster

        # Track pixel savings for directional clipping
        if useDirectional:
            actualPixels = (colEnd - colStart) * (rowEnd - rowStart)
            # Estimate what symmetric clipping would have produced
            pra_w = max(xMaxs) - min(xMins)
            pra_h = max(yMaxs) - min(yMins)
            symCols = int((pra_w + 2 * bufferSize) / cellSize) + 1
            symRows = int((pra_h + 2 * bufferSize) / cellSize) + 1
            symPixels = symCols * symRows
            totalPixelsSaved += max(0, symPixels - actualPixels)

        # Queue for writing
        write_tasks.append((dirName, clippedData, clippedHeader, outputDir))

    # Summary for directional clipping
    if useDirectional:
        log.info(f"Directional clipping applied to {dirClipCount}/{n_groups} groups")
        if totalPixelsSaved > 0:
            savedMB = totalPixelsSaved * 4 / (1024 * 1024)  # float32 = 4 bytes
            log.info(f"Estimated pixel savings: {totalPixelsSaved:,} pixels (~{savedMB:.1f} MB)")

    # Phase 2: Write all clips in parallel (I/O-bound)
    if parallel and len(write_tasks) > 1:
        print(f"Writing {len(write_tasks)} clipped DEMs in parallel (workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_write_clipped_dem, task): task[0] for task in write_tasks}
            
            iterator = tqdm(as_completed(futures), total=len(futures), desc="Writing DEMs", unit="file") if HAS_TQDM else as_completed(futures)
            for future in iterator:
                dirName, extents = future.result()
                groupExtents[dirName] = extents
                log.debug(f"Clipped DEM saved for: {dirName}")
    else:
        # Sequential fallback
        print(f"Writing {len(write_tasks)} clipped DEMs...")
        iterator = tqdm(write_tasks, desc="Writing DEMs", unit="file") if HAS_TQDM else write_tasks
        for task in iterator:
            dirName, extents = _write_clipped_dem(task)
            groupExtents[dirName] = extents
            log.debug(f"Clipped DEM saved for: {dirName}")

    return groupExtents


def clipAndMoveOptionalInput(allSimInputFiles, outputDir, groupExtents):
    """Clip and move ENT and RES files based on group DEM extent.

    Parameters
    ----------
    allSimInputFiles: dict
        With all input information for a com1DFA sim
    outputDir : pathlib.Path
        Path to output directory where clipped files will be saved
    groupExtents : dict
        Dictionary with dirName as key and (xMin, xMax, yMin, yMax) as value,
        containing the DEM clipping extents for each group

    Returns
    -------
    groupFeatures : dict
        Dictionary containing clipped features for each group and type
        {dirName: {'ENT': [...], 'RES': [...]}}
    """
    groupFeatures = {}
    # Process ENT and RES directories
    for dirType in ["ENT", "RES"]:

        if dirType == "ENT":
            if allSimInputFiles["entFile"]:
                shpFile = allSimInputFiles["entFile"]
            else:
                log.info("No entrainment file found")
                continue

        if dirType == "RES":
            if allSimInputFiles["resFile"]:
                shpFile = allSimInputFiles["resFile"]
            else:
                log.info("No resistance file found")
                continue

        # Read shapefile
        fields, fieldNames, properties, geometries, srs = shpConv.readShapefile(shpFile)

        # Process each output directory that has extents
        for entry in outputDir.iterdir():
            if not entry.is_dir() or entry.name not in groupExtents:
                continue

            # Get extent
            xMin, xMax, yMin, yMax = groupExtents[entry.name]
            scenarioBbox = box(xMin, yMin, xMax, yMax)

            # Initialize group in dictionary if not exists
            if entry.name not in groupFeatures:
                groupFeatures[entry.name] = {"ENT": [], "RES": []}

            # Clip geometries with groups DEM extent
            clippedFeatures = []
            for prop, geom in zip(properties, geometries):
                if geom.intersects(scenarioBbox):
                    clippedGeom = geom.intersection(scenarioBbox)
                    if not clippedGeom.is_empty:
                        clippedFeatures.append((prop, clippedGeom))
                        groupFeatures[entry.name][dirType].append(clippedGeom)

            if not clippedFeatures:
                log.debug(f"No {dirType} features intersect with DEM extent for {entry.name}")
                continue

            # Create output directory and save clipped shp
            targetDir = entry / "Inputs" / dirType
            fU.makeADir(targetDir)
            outputPath = targetDir / f"{entry.name}_{dirType}.shp"
            shpConv.writeShapefile(outputPath, fields, fieldNames, clippedFeatures, srs)
            log.debug(f"Clipped {dirType} shapefile saved to: {outputPath}")

    return groupFeatures


def getScenarioGroups(inputShp, fieldNames):
    """Group shapefile records by their scenario attribute.

    Parameters
    ----------
    inputShp : pathlib.Path
        Path to input shapefile
    fieldNames : list
        List of field names in the shapefile

    Returns
    -------
    scenarios: dict
        Dictionary mapping scenario names to lists of shape records
    """
    scenarios = {}
    for shapeRecord in shapefile.Reader(str(inputShp)).iterShapeRecords():
        properties = {k.lower(): v for k, v in zip(fieldNames, shapeRecord.record)}
        scenarioValues = properties.get("scenario", "").split(",")
        for scenario in scenarioValues:
            # Check if scenario value is empty and set flag
            if scenario.strip() == "":
                scenario = "NULL"
            # If scenario is not in scenarios dict, add it
            if scenario not in scenarios:
                scenarios[scenario] = []
            scenarios[scenario].append(shapeRecord)
    return scenarios


def writeScenarioShapefile(outputShp, records, fields, fieldNames, srs):
    """Write a shapefile for a specific scenario.

    Parameters
    ----------
    outputShp : pathlib.Path
        Path where to write the shapefile
    records : list
        List of shape records for this scenario
    fields : list
        List of field definitions
    fieldNames : list
        List of field names
    srs : str
        Spatial reference system string
    """
    # Filter out the scenario attribute
    shapeFeatures = [(dict(zip(fieldNames, record.record)), record.shape) for record in records]
    filteredFields = [field for field in fields if field[0].lower() != "scenario"]
    filteredFieldNames = [name for name in fieldNames if name.lower() != "scenario"]

    # Write the shapefile
    shpConv.writeShapefile(outputShp, filteredFields, filteredFieldNames, shapeFeatures, srs)


def splitByScenarios(dirList, outputDir):
    """Split release areas into separate shapefiles based on their scenario attribute.

    Parameters
    ----------
    dirList: list
        list of dictionaries containing dirName and list of geometries
    outputDir: pathlib.Path object
        path to output directory

    Returns
    -------
    none

    Notes
    -----
    - If a feature has no scenario attribute or it's empty, it will be marked as 'NULL' and grouped together with other 'NULL' features
    - Intermediate shapefiles are deleted or renamed after scenario splitting
    """
    totalInputFiles = 0
    totalScenarioFiles = 0

    # Loop through each folder with progress bar
    n_folders = len(dirList)
    print(f"Splitting {n_folders} folders by scenarios...")
    
    iterator = tqdm(dirList, desc="Splitting scenarios", unit="folder") if HAS_TQDM else dirList
    for folder in iterator:
        inputShp = pathlib.Path(outputDir) / folder["dirName"] / "Inputs" / "REL" / folder["dirName"]
        fields, fieldNames, properties, geometries, srs = shpConv.readShapefile(inputShp)
        totalInputFiles += 1

        # Get the scenario attribute values
        if "scenario" in map(str.lower, fieldNames):
            # Group records by scenario
            scenarios = getScenarioGroups(inputShp, fieldNames)

            # Write a shapefile for each scenario
            for scenario, records in scenarios.items():
                if all(scenario == "NULL" for scenario in scenarios):
                    outputShp = (
                        pathlib.Path(outputDir)
                        / folder["dirName"]
                        / "Inputs"
                        / "REL"
                        / f"{folder['dirName']}_REL"
                    )
                elif scenario == "NULL":
                    outputShp = (
                        pathlib.Path(outputDir)
                        / folder["dirName"]
                        / "Inputs"
                        / "REL"
                        / f"{folder['dirName']}_NULL"
                    )
                else:
                    outputShp = (
                        pathlib.Path(outputDir)
                        / folder["dirName"]
                        / "Inputs"
                        / "REL"
                        / f"{folder['dirName']}_{scenario}"
                    )

                writeScenarioShapefile(outputShp, records, fields, fieldNames, srs)
                totalScenarioFiles += 1

            # Delete the intermediate shapefile
            for ext in [".shp", ".shx", ".dbf", ".prj"]:
                if (inputShp.with_suffix(ext)).exists():
                    (inputShp.with_suffix(ext)).unlink()
        else:
            # If no scenario attribute exists, rename the file (necessary for further processing)
            outputShp = (
                pathlib.Path(outputDir) / folder["dirName"] / "Inputs" / "REL" / f"{folder['dirName']}_REL"
            )
            shapeFeatures = [
                (dict(zip(fieldNames, record.record)), record.shape)
                for record in shapefile.Reader(str(inputShp)).iterShapeRecords()
            ]
            shpConv.writeShapefile(outputShp, fields, fieldNames, shapeFeatures, srs)
            for ext in [".shp", ".shx", ".dbf", ".prj"]:
                if (inputShp.with_suffix(ext)).exists():
                    (inputShp.with_suffix(ext)).unlink()

    if totalScenarioFiles == 0:
        log.info("No 'scenario' attribute or only 'NULL' found in release area shapefiles, continuing")
    else:
        log.info(f"Split '{totalInputFiles}' release area shapefiles into '{totalScenarioFiles}' scenarios")


def writeScenarioReport(dirListGrouped, outputDir):
    """Create a report in txt format listing all scenarios and their associated features.

    Parameters
    ----------
    dirListGrouped : list
        list of dictionaries containing dirName and list of geometries
    outputDir : pathlib.Path
        Path to output directory where the report will be saved

    Returns
    -------
    none
    """
    reportPath = outputDir / "splitInputs_scenarioReport.txt"

    with open(reportPath, "w") as f:
        f.write("SCENARIO REPORT\n")
        f.write("==============\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Process each group and their scenarios
        for group in sorted(dirListGrouped, key=lambda x: x["dirName"].lower()):
            dirName = group["dirName"]
            f.write(f"Group: {dirName}\n")
            f.write("-" * (len(dirName) + 7) + "\n\n")

            relPath = pathlib.Path(outputDir) / dirName / "Inputs" / "REL"
            scenarioFiles = sorted(relPath.glob(f"{dirName}_*.shp"), key=lambda x: x.stem.split("_")[-1])

            if not scenarioFiles:
                f.write("No scenarios found\n\n")
                continue

            # Write release areas for each scenario
            for scenFile in scenarioFiles:
                fields, fieldNames, properties, geometries, _ = shpConv.readShapefile(scenFile)
                scenName = scenFile.stem.split("_")[-1]

                f.write(f"Scenario: {scenName}\n")
                f.write(f"No. of release areas: {len(geometries)}\n")

                if "name" in map(str.lower, fieldNames):  # Handle case sensitivity
                    nameIdx = [i for i, name in enumerate(fieldNames) if name.lower() == "name"][0]
                    with shapefile.Reader(str(scenFile)) as shp:
                        records = sorted(shp.records(), key=lambda x: x[nameIdx].lower())
                        for record in records:
                            f.write(f"- {record[nameIdx]}\n")
                else:
                    for i in range(len(geometries)):
                        f.write(f"- Release Area {i+1}\n")
                f.write("\n")

            # Write total entrainment and resistance areas for the group
            entPath = pathlib.Path(outputDir) / dirName / "Inputs" / "ENT"
            resPath = pathlib.Path(outputDir) / dirName / "Inputs" / "RES"

            entFiles = list(entPath.glob(f"{dirName}_*.shp"))
            if entFiles:
                totalEnt = sum(len(shpConv.readShapefile(ef)[3]) for ef in entFiles)
                if totalEnt > 0:
                    f.write(f"No. of entrainment areas: {totalEnt}\n")

            resFiles = list(resPath.glob(f"{dirName}_*.shp"))
            if resFiles:
                totalRes = sum(len(shpConv.readShapefile(rf)[3]) for rf in resFiles)
                if totalRes > 0:
                    f.write(f"No. of resistance areas: {totalRes}\n")

            f.write("\n")

    log.info(f"Scenario report written to: {reportPath}")


def createReportPlots(dirListGrouped, inputDEM, outputDir, groupExtents, groupFeatures):
    """Write visual reports summarizing the split inputs operation.

    Creates two visual reports in PNG format:
    1. Basic report showing DEM extent and release areas
    2. Optional features report showing DEM extent with entrainment and resistance areas

    Parameters
    ----------
    dirListGrouped : list
        List of dictionaries containing dirName and list of geometries
    inputDEM : pathlib.Path
        Path to input DEM file
    outputDir : pathlib.Path
        Path to output directory where reports will be saved
    groupExtents : dict
        Dictionary with dirName as key and (xMin, xMax, yMin, yMax) as value,
        containing the DEM clipping extents for each group
    groupFeatures : dict
        Dictionary containing clipped features for each group and type

    Returns
    -------
    none
    """
    # Create basic features report
    basicPath = createReportPlot(dirListGrouped, inputDEM, outputDir, groupExtents, groupFeatures, "basic")
    log.info(f"Visual report (basic) written to: {basicPath}")

    # Create optional features report
    optionalPath = createReportPlot(
        dirListGrouped, inputDEM, outputDir, groupExtents, groupFeatures, "optional"
    )
    log.info(f"Visual report (optional) written to: {optionalPath}")
