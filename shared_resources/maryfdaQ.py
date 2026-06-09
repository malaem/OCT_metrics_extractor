"""Scan quality scoring for Topcon FDA files (Mary's QC algorithm).

Implements the quality-assessment logic originally developed in R to flag
low-quality OCT scans.  A scan passes QC if it meets all of the following
criteria (applied per scan type):

- Topcon internal quality score (topQ) above the minimum threshold.
- Mirror position within the expected range for the scan type.
- Fovea-to-disc distance within physiological bounds (Wide/Macula scans).
- Sector thickness values free of NaN runs (configurable strictness).

All input vectors must be in the same row order.  The function returns a list
of row indices that pass QC, or ``999999999999999`` if quality could not be
assessed.

Usage
-----
Called once per scan from ``compute_quality_score`` in ``sector_metrics.py``,
which combines metrics from all extracted layers into a single-row DataFrame
before invoking this function.
Author: Marco Miranda
Date: 28 May 2026"""
## Mary's version to remove low quality fda files

# all vectors need to be in the same order so that in (example) row two of fixation 
#corresponds to topQ presented in row 2, DiscX in row 2 etc...

# Most of the variables needed can be obtained from the TSNIT.csv file generated
# from the OCT data collector program or from the ETDRS for macula.

# Returns the row numbers to keep


# fixation: type of scan "Wide", "Macula", "Disc", etc.
# topQ: topcon quality obtained from fda file (vector).
# discX: Disc position (Auto) in horizontal dimension (vector)(0 to 1).
# discY: Disc position (Auto) in vertical dimension (vector)(0 to 1).
# foveaX: Fovea position (Auto) in horizontal dimension (vector)(0 to 1). 
# foveaY: Fovea position (Auto) in vertical dimension (vector)(0 to 1).
# DA: Disc area (vector). Fix this to any number different from NA, if not present in datav set.
# layerthick: Layer (e.g. Retinal Nerve Fibre Layer) Thickness. Tibble, where each row corresponds
#             to the order in id, and each column contains, for example: RNFL @ Total, Quadrant, 
#             Clockhour and 36 sector for TSNIT scans or full Retina at ETDRS grid for MACULA scans.
#             The algo will automatically differenciate between macular scans and TSNIT scans, so all
#             layers can be provided at once.
# mirroPos: Mirror position. It can be obtained from the AxialLengthEstimator program (vector).
# scanXdim: Dimension of scan in horizontal position (mm). Defaults to 12 (12 x 9 wide scan). Can be a single number or a vector of numbers.
# scanYDim: Dimension of scan in the vertical position (mm). Defaults to 9 (12 x 9 wide scan). Can be a single number or a vector of numbers.
# criteria: more or less stringent ("more", "less"). "less" allows some NAs - less than 95% of the data set and less than 3 consecutive.
#           "more" does not allow any NA is layerthick. Default is "more"


#from shared_resources.grid_diameter import grid_diameter
#from shared_resources.cpRNFLsectorAverage import cpRNFLsectorAverage

#import rle
import pandas as pd
import numpy as np



# create function to detect rows with a run of NAs longer than na_run_limit
def has_long_na_run_vectorized(df, columns, limit):
    """Return a boolean Series: True if each row has no consecutive-NaN run ≥ *limit*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check for consecutive NaN runs.
    columns : list[str]
        Column names to evaluate (column order matters — rolling is along axis 1).
    limit : int
        Maximum allowed consecutive-NaN run length (exclusive).  A row with
        ``limit`` or more consecutive NaNs in *columns* returns ``False``.

    Returns
    -------
    pd.Series of bool
        ``True`` for rows that pass (run length < limit), ``False`` for rows
        that have a run of ``limit`` or more consecutive NaNs.
    """
    is_na_df = df[columns].isna().astype(int)
    # DataFrame.rolling(axis=1) was removed in pandas 2.0.
    # Transpose so we roll along rows (each row becomes a column), then transpose back.
    max_consecutive = is_na_df.T.rolling(window=limit).sum().T.max(axis=1)
    return max_consecutive < limit


def maryfdaQ(fixation, topQ, scan_size, disc_center, fovea, DA, layerthick, mirrorPos, f2d_distance, f2d_angle, criteria = "more"):
    """Assess scan quality and return the indices of scans that pass QC.

    Parameters
    ----------
    fixation : str or array-like of str
        Scan type(s): ``"Wide"``, ``"Disc"``, ``"Macula"``, or ``"External"``.
    topQ : float or array-like of float
        Topcon internal quality score per scan (higher = better).
    scan_size : list of [x_mm, y_mm] or array-like
        Scan dimensions in millimetres ``[width, height]``.
    disc_center : list of [x_prop, y_prop] or array-like
        Optic disc centre coordinates as proportions in [0, 1].
    fovea : list of [x_prop, y_prop] or array-like
        Fovea centre coordinates as proportions in [0, 1].
    DA : float or array-like of float
        Disc area (mm²).  Pass any non-NA value if not available.
    layerthick : pd.DataFrame
        Sector thickness metrics, one row per scan.  Columns are the layer
        sector labels (e.g. ``'Total'``, ``'4_T'``, ``'ETDRS_Center'`` …).
    mirrorPos : int or array-like of int
        Mirror position from the FDA file (used for axial-length thresholding).
    f2d_distance : float or array-like of float
        Fovea-to-disc distance in mm.
    f2d_angle : float or array-like of float
        Fovea-to-disc angle in degrees.
    criteria : {"more", "less"}, default "more"
        QC strictness.
        - ``"more"`` : no NaN values allowed in ``layerthick``.
        - ``"less"`` : up to 5 % NaN values and fewer than 3 consecutive NaNs
          are tolerated.

    Returns
    -------
    list[int] or int
        List of zero-based row indices that pass all QC filters, or the
        sentinel value ``999999999999999`` when quality cannot be assessed
        (e.g. unrecognised scan type or insufficient data).
    """
    #do some convertions needed
    fixation = pd.Series(fixation)
    # None disc area must become NaN, not an empty Series. pd.Series(None) in pandas
    # creates an empty Series, causing DA.notna() to be empty and all Wide/Disc
    # quality conditions to evaluate to empty — incorrectly producing "Fail" output.
    DA = pd.Series([np.nan if DA is None else DA])
    #layerthick = pd.Series(layerthick)
    mirrorPos = pd.Series(mirrorPos)
    topQ = pd.Series(topQ)
    # None f2d values (non-Wide scans, or Wide scans where the calculation failed)
    # must become NaN so that the boolean conditions below evaluate to False rather
    # than raising TypeError: '>' not supported between 'NoneType' and 'int'.
    f2d_distance = pd.Series([np.nan if f2d_distance is None else f2d_distance])
    f2d_angle    = pd.Series([np.nan if f2d_angle    is None else f2d_angle])
    # np.atleast_1d normalises scalars (single-scan call) and arrays (batch call)
    # to the same 1-D form so that subscript access [index_ext[i]] works in both cases.
    scanXdim = pd.Series(np.atleast_1d(scan_size[0]))
    scanYdim = pd.Series(np.atleast_1d(scan_size[1]))
    discX    = pd.Series(np.atleast_1d(disc_center[0]))
    discY    = pd.Series(np.atleast_1d(disc_center[1]))
    foveaX   = pd.Series(np.atleast_1d(fovea[0]))
    foveaY   = pd.Series(np.atleast_1d(fovea[1]))

    if criteria == "less":
        # set proportion of NAs per row and maximum allowed run length of consecutive NAs
        prop_threshold = 0.95
        na_run_limit = 3
    
    #it crashes here
    index_wide = np.where(fixation == "Wide")[0]
    index_disc = np.where(fixation == "Disc")[0]
    index_mac = np.where(fixation == "Macula")[0]
    index_ext = np.where(fixation == "External")[0]
    index_ext_wide = pd.Series([pd.NA], dtype = 'Int64')
    index_ext_disc = pd.Series([pd.NA], dtype = 'Int64')
    index_ext_unrec = pd.Series([pd.NA], dtype = 'Int64')
    index_ext_mac = pd.Series([pd.NA], dtype = 'Int64')

    # Find type of scan for each external.
    for i in range(len(index_ext)):
        
        if scanXdim[index_ext[i]] == 12 and scanYdim[index_ext[i]] == 9:
        
            index_ext_wide = pd.concat([index_ext_wide, pd.Series([index_ext[i]], dtype='Int64')], ignore_index=True)
        
        elif scanXdim[index_ext[i]] == 6 and scanYdim[index_ext[i]] == 6 and discX[index_ext[i]] == 0.5 and discY[index_ext[i]] == 0.5 and foveaX[index_ext[i]] != 0.5 and foveaY[index_ext[i]] != 0.5:
        
            index_ext_mac = pd.concat([index_ext_mac, pd.Series([index_ext[i]], dtype='Int64')], ignore_index=True)
        
        elif scanXdim[index_ext[i]] == 6 and scanYdim[index_ext[i]] == 6 and discX[index_ext[i]] != 0.5 and discY[index_ext[i]] != 0.5 and foveaX[index_ext[i]] == 0.5 and foveaY[index_ext[i]] == 0.5:
        
            index_ext_disc = pd.concat([index_ext_disc, pd.Series([index_ext[i]], dtype='Int64')], ignore_index=True)
        
        else:
        
            print(f"do no recognise scan in row {index_ext[i]}, it will only look at TopQ and mirror position for quality assessment")
            index_ext_unrec = pd.concat([index_ext_unrec, pd.Series([index_ext[i]], dtype='Int64')], ignore_index=True)
        
    
    # remove NAs
    index_ext_wide = index_ext_wide.dropna()
    index_ext_disc = index_ext_disc.dropna()
    index_ext_unrec = index_ext_unrec.dropna()
    index_ext_mac = index_ext_mac.dropna()
    
    
    # --- Wide scans ----
    if index_wide.size != 0:
        # Build column list dynamically based on available layer data
        # Check for cpRNFL, GCL+/GCL++, and Retina columns
        cprnfl_cols = ["Total", "4_T", "4_S", "4_N", "4_I", "6_T", "6_TS",         
                    "6_NS", "6_N", "6_NI", "6_TI", "12_T", "12_TS", "12_ST",        
                    "12_S", "12_SN", "12_NS", "12_N", "12_NI", "12_IN", "12_I",         
                    "12_IT", "12_TI", "36_01", "36_02", "36_03", "36_04", "36_05",        
                    "36_06", "36_07", "36_08", "36_09", "36_10", "36_11", "36_12",        
                    "36_13", "36_14", "36_15", "36_16", "36_17", "36_18", "36_19",        
                    "36_20", "36_21", "36_22", "36_23", "36_24", "36_25", "36_26",        
                    "36_27", "36_28", "36_29", "36_30", "36_31", "36_32", "36_33",        
                    "36_34", "36_35", "36_36"
        ]
        
        macula6_cols = ["TS", "S", "NS", "NI", "I", "TI", "Total"]
        
        etdrs_cols = ["ETDRS_Center", "ETDRS_In_T", "ETDRS_In_S", "ETDRS_In_N",
                      "ETDRS_In_I", "ETDRS_Out_T", "ETDRS_Out_S", "ETDRS_Out_N",
                      "ETDRS_Out_I", "average_thick"]
        # center_thick and total_vol intentionally excluded: currently always None in output
        
        # Determine which columns are available (all must be present for a layer group to be checked)
        has_cprnfl = all(col in layerthick.columns for col in cprnfl_cols)
        has_macula6 = all(col in layerthick.columns for col in macula6_cols)
        has_etdrs = all(col in layerthick.columns for col in etdrs_cols)
        
        # Build combined column list from all available layers
        columns_to_check = []
        if has_cprnfl:
            columns_to_check.extend(cprnfl_cols)
        if has_macula6:
            columns_to_check.extend(macula6_cols)
        if has_etdrs:
            columns_to_check.extend(etdrs_cols)
        
        # Also include any layer-suffixed duplicate columns (e.g. 'Total__GCL+',
        # 'Total__GCL++') added by compute_quality_score when multiple layers share
        # a column name. Without this, a NA in the second layer's Total would be
        # silently ignored by the notna() check below.
        columns_to_check += [c for c in layerthick.columns if '__' in c and c.split('__')[0] in columns_to_check]
        
        # Only proceed if we have at least one set of columns
        if columns_to_check:
            # Inclusion criteria for Wide scans (check all available layers)
            if criteria == "more":
                combined_condition = (
                    (f2d_distance > 3) & (f2d_distance < 6) &
                    (abs((f2d_angle +180 ) % 360-180) <= 21) &
                    (discX > 0.2) & (discX < 1 - 0.2) &
                    (discY > 0.2) & (discY < 1 - 0.2) &
                    (foveaX > 0.25) & (foveaX < 1 - 0.25) &
                    (foveaY > 0.25) & (foveaY < 1 - 0.25) &
                    (DA.notna()) & (DA < 6.5) &
                    (topQ > 18) &
                    (layerthick[columns_to_check].notna().all(axis=1)) & # find rows where any column is NA
                    (mirrorPos >= 538) & (mirrorPos <= 4000)
                )  
            elif criteria == "less":
                combined_condition = (
                    (f2d_distance > 3) & (f2d_distance < 6) &
                    (abs((f2d_angle +180 ) % 360-180) <= 21) &
                    (discX > 0.2) & (discX < 1 - 0.2) &
                    (discY > 0.2) & (discY < 1 - 0.2) &
                    (foveaX > 0.25) & (foveaX < 1 - 0.25) &
                    (foveaY > 0.25) & (foveaY < 1 - 0.25) &
                    (DA.notna()) & (DA < 6.5) &
                    (topQ > 18) &
                    (layerthick[columns_to_check].notna().mean(axis=1) >= prop_threshold) & # find rows where proportion of NAs is < than a 5%
                    (has_long_na_run_vectorized(layerthick, columns_to_check, na_run_limit)) & # find rows where number of consecutive NAs is < than 3
                    (mirrorPos >= 538) & (mirrorPos <= 4000)
                )
            
            index = combined_condition[combined_condition].index.tolist()
            index_wide_set = set(index_wide)
            index_wide = [i for i in index if i in index_wide_set]
    
    # --- Disc scans ----
    # NOTE: Disc scans support cpRNFL only (LAYER_FIXATION_COMPAT enforces this
    # upstream). The hardcoded column list below assumes cpRNFL sector columns
    # are present. If cpRNFL was not selected the KeyError is caught by the
    # compute_quality_score caller, which returns None (no QC score). This is
    # intentional — requesting a non-cpRNFL layer on a Disc scan is blocked
    # before this function is called.
    if index_disc.size != 0:
        columns = ["Total", "4_T", "4_S", "4_N", "4_I", "6_T", "6_TS",         
                    "6_NS", "6_N", "6_NI", "6_TI", "12_T", "12_TS", "12_ST",        
                    "12_S", "12_SN", "12_NS", "12_N", "12_NI", "12_IN", "12_I",         
                    "12_IT", "12_TI", "36_01", "36_02", "36_03", "36_04", "36_05",        
                    "36_06", "36_07", "36_08", "36_09", "36_10", "36_11", "36_12",        
                    "36_13", "36_14", "36_15", "36_16", "36_17", "36_18", "36_19",        
                    "36_20", "36_21", "36_22", "36_23", "36_24", "36_25", "36_26",        
                    "36_27", "36_28", "36_29", "36_30", "36_31", "36_32", "36_33",        
                    "36_34", "36_35", "36_36"
        ]
            
        # Inclusion criteria
        if criteria == "more":
            combined_condition = (
                (discX > 0.3) & (discX < 1 - 0.3) &
                (discY > 0.3) & (discY < 1 - 0.3) &
                (DA.notna()) & (DA < 6.5) &
                (topQ > 18) &
                (layerthick[columns].notna().all(axis=1)) & # find rows where any column is NA
                (mirrorPos >= 538) & (mirrorPos <= 4000)
            )
        elif criteria == "less":
            combined_condition = (
                (discX > 0.3) & (discX < 1 - 0.3) &
                (discY > 0.3) & (discY < 1 - 0.3) &
                (DA.notna()) & (DA < 6.5) &
                (topQ > 18) &
                (layerthick[columns].notna().mean(axis=1) >= prop_threshold) & # find rows where proportion of NAs is < than a 5%
                (has_long_na_run_vectorized(layerthick, columns, na_run_limit)) & # find rows where number of consecutive NAs is < than 3
                (mirrorPos >= 538) & (mirrorPos <= 4000)
            )
            
        index = combined_condition[combined_condition].index.tolist()
        index_disc_set = set(index_disc)
        index_disc = [i for i in index if i in index_disc_set]
        
    # --- Macula scans ---- 
    if index_mac.size != 0:
        macula6_cols = ["TS", "S", "NS", "NI", "I", "TI", "Total"]
        etdrs_cols = ["ETDRS_Center", "ETDRS_In_T", "ETDRS_In_S", "ETDRS_In_N",
                      "ETDRS_In_I", "ETDRS_Out_T", "ETDRS_Out_S", "ETDRS_Out_N",
                      "ETDRS_Out_I", "average_thick"]
        # center_thick and total_vol intentionally excluded: currently always None in output

        has_macula6 = all(col in layerthick.columns for col in macula6_cols)
        has_etdrs = all(col in layerthick.columns for col in etdrs_cols)

        columns = []
        if has_macula6:
            columns.extend(macula6_cols)
        if has_etdrs:
            columns.extend(etdrs_cols)
        
        # Also include layer-suffixed duplicate columns (see Wide section comment).
        columns += [c for c in layerthick.columns if '__' in c and c.split('__')[0] in columns]

        if not columns:
            # No recognised layer columns — skip quality check for Macula scans
            index_mac = []
        else:
            if criteria == "more":
                combined_condition = (
                    (foveaX > 0.3) & (foveaX < 1 - 0.3) &
                    (foveaY > 0.3) & (foveaY < 1 - 0.3) &
                    (topQ > 18) &
                    (layerthick[columns].notna().all(axis=1)) &
                    (mirrorPos >= 538) & (mirrorPos <= 4000)
                )
            elif criteria == "less":
                combined_condition = (
                    (foveaX > 0.3) & (foveaX < 1 - 0.3) &
                    (foveaY > 0.3) & (foveaY < 1 - 0.3) &
                    (topQ > 18) &
                    (layerthick[columns].notna().mean(axis=1) >= prop_threshold) &
                    (has_long_na_run_vectorized(layerthick, columns, na_run_limit)) &
                    (mirrorPos >= 538) & (mirrorPos <= 4000)
                )

            index = combined_condition[combined_condition].index.tolist()
            index_mac_set = set(index_mac)
            index_mac = [i for i in index if i in index_mac_set]
    
    
    # --- External scans ----
    #---- Mind due the below may only be ok for the purpose of axial length
    #analysis. For other criteria one may want to remove these scans ------ 
    

    # --- Wide scans External ----
    if index_ext_wide.size != 0:
    # Inclusion criteria
        combined_condition = (
            (f2d_distance > 3) & (f2d_distance < 6) &
            (abs((f2d_angle +180 ) % 360-180) <= 21) &
            (discX > 0.2) & (discX < 1 - 0.2) &
            (discY > 0.2) & (discY < 1 - 0.2) &
            (foveaX > 0.25) & (foveaX < 1 - 0.25) &
            (foveaY > 0.25) & (foveaY < 1 - 0.25) &
            (topQ > 18) &
            (mirrorPos >= 538) & (mirrorPos <= 4000)
        )
        index = combined_condition[combined_condition].index.tolist()
        index_ext_wide_set = set(index_ext_wide)
        index_ext_wide = [i for i in index if i in index_ext_wide_set]

    # --- Disc scans External ----  
    
    if index_ext_disc.size != 0:
        # Inclusion criteria
        combined_condition = (
            (discX > 0.3) & (discX < 1 - 0.3) &
            (discY > 0.3) & (discY < 1 - 0.3) &
            (topQ > 18) &
            (mirrorPos >= 538) & (mirrorPos <= 4000)
        )
        index = combined_condition[combined_condition].index.tolist()
        index_ext_disc_set = set(index_ext_disc)
        index_ext_disc = [i for i in index if i in index_ext_disc_set]    
    
    
    # --- Macula scans External ---- 
    if index_ext_mac.size != 0:
        # Inclusion criteria
        combined_condition = (
            (foveaX > 0.3) & (foveaX < 1 - 0.3) &
            (foveaY > 0.3) & (foveaY < 1 - 0.3) &
            (topQ > 18) &
            (mirrorPos >= 538) & (mirrorPos <= 4000)
        )
        index = combined_condition[combined_condition].index.tolist()
        index_ext_mac_set = set(index_ext_mac)
        index_ext_mac = [i for i in index if i in index_ext_mac_set] 
    

    # --- Unrecognised scans External ---- 
    
    if index_ext_unrec.size != 0:
        # Inclusion criteria
        combined_condition = (
            (foveaX > 0.2) & (foveaX < 1 - 0.2) &
            (foveaY > 0.2) & (foveaY < 1 - 0.2) &
            (topQ > 18) &
            (mirrorPos >= 538) & (mirrorPos <= 4000)
        )
        index = combined_condition[combined_condition].index.tolist()
        index_ext_unrec_set = set(index_ext_unrec)
        index_ext_unrec = [i for i in index if i in index_ext_unrec_set]
    
    # ---- final list of acceptable scans ----

    index = [index_wide, index_disc, index_mac, index_ext_wide, index_ext_disc,
            index_ext_mac, index_ext_unrec]
    index = [item for sublist in index if len(sublist) > 0 for item in sublist]
    
    return index