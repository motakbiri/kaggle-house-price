#import libraries
import pandas as pd
import numpy as np
from utils import factorize ,refactor , lasso , gradient_boosting ,random_forest
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler


#load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#------------
#cleaning Data
#------------

#drop more than 4000 m^2 houses
train.drop(train[train["GrLivArea"] > 4000].index,inplace =True)

#fill row 666 of test data with median
test.loc[666, "GarageQual"] = "TA"
test.loc[666, "GarageCond"] = "TA"
test.loc[666, "GarageFinish"] = "Unf"
test.loc[666, "GarageYrBlt"] = "1980"

#1116 row only has GarageType in garage information , fill this with NAN
test.loc[1116,"GarageType"] = np.nan

lot_frontage_by_neighborhood = train["LotFrontage"].groupby(train["Neighborhood"])
#combine features
def combine(dataFrame):
    all = pd.DataFrame(index=dataFrame.index)

    all["LotFrontage"] = dataFrame["LotFrontage"]
    for key, group in lot_frontage_by_neighborhood:
        idx = (dataFrame["Neighborhood"] == key) & (dataFrame["LotFrontage"].isnull())
        all.loc[idx, "LotFrontage"] = group.median()

    all["LotArea"] = dataFrame["LotArea"]

    all["MasVnrArea"] = dataFrame["MasVnrArea"]
    all["MasVnrArea"].fillna(0, inplace=True)

    all["BsmtFinSF1"] = dataFrame["BsmtFinSF1"]
    all["BsmtFinSF1"].fillna(0, inplace=True)

    all["BsmtFinSF2"] = dataFrame["BsmtFinSF2"]
    all["BsmtFinSF2"].fillna(0, inplace=True)

    all["BsmtUnfSF"] = dataFrame["BsmtUnfSF"]
    all["BsmtUnfSF"].fillna(0, inplace=True)

    all["TotalBsmtSF"] = dataFrame["TotalBsmtSF"]
    all["TotalBsmtSF"].fillna(0, inplace=True)

    all["1stFlrSF"] = dataFrame["1stFlrSF"]
    all["2ndFlrSF"] = dataFrame["2ndFlrSF"]
    all["GrLivArea"] = dataFrame["GrLivArea"]

    all["GarageArea"] = dataFrame["GarageArea"]
    all["GarageArea"].fillna(0, inplace=True)

    all["WoodDeckSF"] = dataFrame["WoodDeckSF"]
    all["OpenPorchSF"] = dataFrame["OpenPorchSF"]
    all["EnclosedPorch"] = dataFrame["EnclosedPorch"]
    all["3SsnPorch"] = dataFrame["3SsnPorch"]
    all["ScreenPorch"] = dataFrame["ScreenPorch"]

    all["BsmtFullBath"] = dataFrame["BsmtFullBath"]
    all["BsmtFullBath"].fillna(0, inplace=True)

    all["BsmtHalfBath"] = dataFrame["BsmtHalfBath"]
    all["BsmtHalfBath"].fillna(0, inplace=True)

    all["FullBath"] = dataFrame["FullBath"]
    all["HalfBath"] = dataFrame["HalfBath"]
    all["BedroomAbvGr"] = dataFrame["BedroomAbvGr"]
    all["KitchenAbvGr"] = dataFrame["KitchenAbvGr"]
    all["TotRmsAbvGrd"] = dataFrame["TotRmsAbvGrd"]
    all["Fireplaces"] = dataFrame["Fireplaces"]

    all["GarageCars"] = dataFrame["GarageCars"]
    all["GarageCars"].fillna(0, inplace=True)

    all["CentralAir"] = (dataFrame["CentralAir"] == "Y") * 1.0

    all["OverallQual"] = dataFrame["OverallQual"]
    all["OverallCond"] = dataFrame["OverallCond"]

    # Quality measurements are stored as text but we can convert them to
    # numbers where a higher number means higher quality.

    qual_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    all["ExterQual"] = dataFrame["ExterQual"].map(qual_dict).astype(int)
    all["ExterCond"] = dataFrame["ExterCond"].map(qual_dict).astype(int)
    all["BsmtQual"] = dataFrame["BsmtQual"].map(qual_dict).astype(int)
    all["BsmtCond"] = dataFrame["BsmtCond"].map(qual_dict).astype(int)
    all["HeatingQC"] = dataFrame["HeatingQC"].map(qual_dict).astype(int)
    all["KitchenQual"] = dataFrame["KitchenQual"].map(qual_dict).astype(int)
    all["FireplaceQu"] = dataFrame["FireplaceQu"].map(qual_dict).astype(int)
    all["GarageQual"] = dataFrame["GarageQual"].map(qual_dict).astype(int)
    all["GarageCond"] = dataFrame["GarageCond"].map(qual_dict).astype(int)

    all["BsmtExposure"] = dataFrame["BsmtExposure"].map(
        {None: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)

    bsmt_fin_dict = {None: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    all["BsmtFinType1"] = dataFrame["BsmtFinType1"].map(bsmt_fin_dict).astype(int)
    all["BsmtFinType2"] = dataFrame["BsmtFinType2"].map(bsmt_fin_dict).astype(int)

    all["Functional"] = dataFrame["Functional"].map(
        {None: 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4,
         "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)

    all["GarageFinish"] = dataFrame["GarageFinish"].map(
        {None: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)

    all["Fence"] = dataFrame["Fence"].map(
        {None: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)

    all["YearBuilt"] = dataFrame["YearBuilt"]
    all["YearRemodAdd"] = dataFrame["YearRemodAdd"]

    all["GarageYrBlt"] = dataFrame["GarageYrBlt"]
    all["GarageYrBlt"].fillna(0.0, inplace=True)

    all["MoSold"] = dataFrame["MoSold"]
    all["YrSold"] = dataFrame["YrSold"]

    all["LowQualFinSF"] = dataFrame["LowQualFinSF"]
    all["MiscVal"] = dataFrame["MiscVal"]

    all["PoolQC"] = dataFrame["PoolQC"].map(qual_dict).astype(int)

    all["PoolArea"] = dataFrame["PoolArea"]
    all["PoolArea"].fillna(0, inplace=True)

    # Add categorical features as numbers too. It seems to help a bit.
    factorize(dataFrame, all, "MSSubClass")
    factorize(dataFrame, all, "MSZoning", "RL")
    factorize(dataFrame, all, "LotConfig")
    factorize(dataFrame, all, "Neighborhood")
    factorize(dataFrame, all, "Condition1")
    factorize(dataFrame, all, "BldgType")
    factorize(dataFrame, all, "HouseStyle")
    factorize(dataFrame, all, "RoofStyle")
    factorize(dataFrame, all, "Exterior1st", "Other")
    factorize(dataFrame, all, "Exterior2nd", "Other")
    factorize(dataFrame, all, "MasVnrType", "None")
    factorize(dataFrame, all, "Foundation")
    factorize(dataFrame, all, "SaleType", "Oth")
    factorize(dataFrame, all, "SaleCondition")

    # IR2 and IR3 don't appear that often, so just make a distinction
    # between regular and irregular.
    all["IsRegularLotShape"] = (dataFrame["LotShape"] == "Reg") * 1

    # Most properties are level; bin the other possibilities together
    # as "not level".
    all["IsLandLevel"] = (dataFrame["LandContour"] == "Lvl") * 1

    # Most land slopes are gentle; treat the others as "not gentle".
    all["IsLandSlopeGentle"] = (dataFrame["LandSlope"] == "Gtl") * 1

    # Most properties use standard circuit breakers.
    all["IsElectricalSBrkr"] = (dataFrame["Electrical"] == "SBrkr") * 1

    # About 2/3rd have an attached garage.
    all["IsGarageDetached"] = (dataFrame["GarageType"] == "Detchd") * 1

    # Most have a paved drive. Treat dirt/gravel and partial pavement
    # as "not paved".
    all["IsPavedDrive"] = (dataFrame["PavedDrive"] == "Y") * 1

    # The only interesting "misc. feature" is the presence of a shed.
    all["HasShed"] = (dataFrame["MiscFeature"] == "Shed") * 1.

    # If YearRemodAdd != YearBuilt, then a remodeling took place at some point.
    all["Remodeled"] = (all["YearRemodAdd"] != all["YearBuilt"]) * 1

    # Did a remodeling happen in the year the house was sold?
    all["RecentRemodel"] = (all["YearRemodAdd"] == all["YrSold"]) * 1

    # Was this house sold in the year it was built?
    all["VeryNewHouse"] = (all["YearBuilt"] == all["YrSold"]) * 1

    all["Has2ndFloor"] = (all["2ndFlrSF"] == 0) * 1
    all["HasMasVnr"] = (all["MasVnrArea"] == 0) * 1
    all["HasWoodDeck"] = (all["WoodDeckSF"] == 0) * 1
    all["HasOpenPorch"] = (all["OpenPorchSF"] == 0) * 1
    all["HasEnclosedPorch"] = (all["EnclosedPorch"] == 0) * 1
    all["Has3SsnPorch"] = (all["3SsnPorch"] == 0) * 1
    all["HasScreenPorch"] = (all["ScreenPorch"] == 0) * 1

    # Months with the largest number of deals may be significant.
    all["HighSeason"] = dataFrame["MoSold"].replace(
        {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})

    all["NewerDwelling"] = dataFrame["MSSubClass"].replace(
        {20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
         90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})

    all.loc[dataFrame.Neighborhood == 'NridgHt', "Neighborhood_Good"] = 1
    all.loc[dataFrame.Neighborhood == 'Crawfor', "Neighborhood_Good"] = 1
    all.loc[dataFrame.Neighborhood == 'StoneBr', "Neighborhood_Good"] = 1
    all.loc[dataFrame.Neighborhood == 'Somerst', "Neighborhood_Good"] = 1
    all.loc[dataFrame.Neighborhood == 'NoRidge', "Neighborhood_Good"] = 1
    all["Neighborhood_Good"].fillna(0, inplace=True)

    all["SaleCondition_PriceDown"] = dataFrame.SaleCondition.replace(
        {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})

    # House completed before sale or not
    all["BoughtOffPlan"] = dataFrame.SaleCondition.replace(
        {"Abnorml": 0, "Alloca": 0, "AdjLand": 0, "Family": 0, "Normal": 0, "Partial": 1})

    all["BadHeating"] = dataFrame.HeatingQC.replace(
        {'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})

    area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea']
    all["TotalArea"] = all[area_cols].sum(axis=1)

    all["TotalArea1st2nd"] = all["1stFlrSF"] + all["2ndFlrSF"]

    all["Age"] = 2010 - all["YearBuilt"]
    all["TimeSinceSold"] = 2010 - all["YrSold"]

    all["SeasonSold"] = all["MoSold"].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                                                 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}).astype(int)

    all["YearsSinceRemodel"] = all["YrSold"] - all["YearRemodAdd"]

    # Simplifications of existing features into bad/average/good.
    all["SimplOverallQual"] = all.OverallQual.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3})
    all["SimplOverallCond"] = all.OverallCond.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3})
    all["SimplPoolQC"] = all.PoolQC.replace(
        {1: 1, 2: 1, 3: 2, 4: 2})
    all["SimplGarageCond"] = all.GarageCond.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    all["SimplGarageQual"] = all.GarageQual.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    all["SimplFireplaceQu"] = all.FireplaceQu.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    all["SimplFireplaceQu"] = all.FireplaceQu.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    all["SimplFunctional"] = all.Functional.replace(
        {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4})
    all["SimplKitchenQual"] = all.KitchenQual.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    all["SimplHeatingQC"] = all.HeatingQC.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    all["SimplBsmtFinType1"] = all.BsmtFinType1.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
    all["SimplBsmtFinType2"] = all.BsmtFinType2.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
    all["SimplBsmtCond"] = all.BsmtCond.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    all["SimplBsmtQual"] = all.BsmtQual.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    all["SimplExterCond"] = all.ExterCond.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
    all["SimplExterQual"] = all.ExterQual.replace(
        {1: 1, 2: 1, 3: 1, 4: 2, 5: 2})

    # Bin by neighborhood (a little arbitrarily). Values were computed by:
    # train_dataFrame["SalePrice"].groupby(train_dataFrame["Neighborhood"]).median().sort_values()
    neighborhood_map = {
        "MeadowV": 0,  # 88000
        "IDOTRR": 1,  # 103000
        "BrDale": 1,  # 106000
        "OldTown": 1,  # 119000
        "Edwards": 1,  # 119500
        "BrkSide": 1,  # 124300
        "Sawyer": 1,  # 135000
        "Blueste": 1,  # 137500
        "SWISU": 2,  # 139500
        "NAmes": 2,  # 140000
        "NPkVill": 2,  # 146000
        "Mitchel": 2,  # 153500
        "SawyerW": 2,  # 179900
        "Gilbert": 2,  # 181000
        "NWAmes": 2,  # 182900
        "Blmngtn": 2,  # 191000
        "CollgCr": 2,  # 197200
        "ClearCr": 3,  # 200250
        "Crawfor": 3,  # 200624
        "Veenker": 3,  # 218000
        "Somerst": 3,  # 225500
        "Timber": 3,  # 228475
        "StoneBr": 4,  # 278000
        "NoRidge": 4,  # 290000
        "NridgHt": 4,  # 315000
    }

    all["NeighborhoodBin"] = dataFrame["Neighborhood"].map(neighborhood_map)
    return all

combined_train = combine(train)
combined_test = combine(test)

#store a copy of NeighborhoodBin for next usage after scaling
neighborhood_bin_train = pd.DataFrame(index=train.index)
neighborhood_bin_train["NeighborhoodBin"] = combined_train["NeighborhoodBin"]
neighborhood_bin_test = pd.DataFrame(index=test.index)
neighborhood_bin_test["NeighborhoodBin"] = combined_test["NeighborhoodBin"]

#find numeric features
numeric_features = combined_train.dtypes[combined_train.dtypes != "object"].index

#normalize data
skewed = combined_train[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index

combined_train[skewed] = np.log1p(combined_train[skewed])
combined_test[skewed] = np.log1p(combined_test[skewed])

#scaling data
scaler = StandardScaler()
scaler.fit(combined_train[numeric_features])

scaled = scaler.transform(combined_train[numeric_features])
for i, col in enumerate(numeric_features):
    combined_train[col] = scaled[:, i]

scaled = scaler.transform(combined_test[numeric_features])
for i, col in enumerate(numeric_features):
    combined_test[col] = scaled[:, i]

#Convert categorical features to numeric with one hot algorithm and encoding
def convert_and_combine(dataFrame):
    refactored_data_frame = pd.DataFrame(index=dataFrame.index)

    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "MSSubClass", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "MSZoning", "RL")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "LotConfig", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "Neighborhood", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "Condition1", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "BldgType", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "HouseStyle", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "RoofStyle", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "Exterior1st", "VinylSd")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "Exterior2nd", "VinylSd")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "Foundation", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "SaleType", "WD")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "SaleCondition", "Normal")

    # Fill in missing MasVnrType for rows that do have a MasVnrArea.
    temp_dataFrame = dataFrame[["MasVnrType", "MasVnrArea"]].copy()
    idx = (dataFrame["MasVnrArea"] != 0) & ((dataFrame["MasVnrType"] == "None") | (dataFrame["MasVnrType"].isnull()))
    temp_dataFrame.loc[idx, "MasVnrType"] = "BrkFace"
    refactored_data_frame = refactor(refactored_data_frame, temp_dataFrame, "MasVnrType", "None")

    # Also add the booleans from calc_dataFrame as dummy variables.
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "LotShape", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "LandContour", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "LandSlope", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "Electrical", "SBrkr")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "GarageType", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "PavedDrive", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "MiscFeature", "None")

    # Features we can probably ignore (but want to include anyway to see
    # if they make any positive difference).
    # Definitely ignoring Utilities: all records are "AllPub", except for
    # one "NoSeWa" in the train set and 2 NA in the test set.
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "Street", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "Alley", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "Condition2", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "RoofMatl", None)
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "Heating", None)

    # I have these as numerical variables too.
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "ExterQual", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "ExterCond", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "BsmtQual", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "BsmtCond", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "HeatingQC", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "KitchenQual", "TA")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "FireplaceQu", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "GarageQual", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "GarageCond", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "PoolQC", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "BsmtExposure", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "BsmtFinType1", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "BsmtFinType2", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "Functional", "Typ")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "GarageFinish", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "Fence", "None")
    refactored_data_frame = refactor(refactored_data_frame, dataFrame, "MoSold", None)

    # Divide up the years between 1871 and 2010 in slices of 20 years.
    year_map = pd.concat(
        pd.Series("YearBin" + str(i + 1), index=range(1871 + i * 20, 1891 + i * 20)) for i in range(0, 7))

    yearbin_dataFrame = pd.DataFrame(index=dataFrame.index)
    yearbin_dataFrame["GarageYrBltBin"] = dataFrame.GarageYrBlt.map(year_map)
    yearbin_dataFrame["GarageYrBltBin"].fillna("NoGarage", inplace=True)

    yearbin_dataFrame["YearBuiltBin"] = dataFrame.YearBuilt.map(year_map)
    yearbin_dataFrame["YearRemodAddBin"] = dataFrame.YearRemodAdd.map(year_map)

    refactored_data_frame = refactor(refactored_data_frame, yearbin_dataFrame, "GarageYrBltBin", None)
    refactored_data_frame = refactor(refactored_data_frame, yearbin_dataFrame, "YearBuiltBin", None)
    refactored_data_frame = refactor(refactored_data_frame, yearbin_dataFrame, "YearRemodAddBin", None)

    return refactored_data_frame


#apply on data frames
refactored_data_frame = convert_and_combine(train)
refactored_data_frame = refactor(refactored_data_frame,neighborhood_bin_train,"NeighborhoodBin", None)
combined_train = combined_train.join(refactored_data_frame)


#is not available in test
drop_columns = [
    "_Exterior1st_ImStucc", "_Exterior1st_Stone",
    "_Exterior2nd_Other", "_HouseStyle_2.5Fin",
    "_RoofMatl_Membran", "_RoofMatl_Metal", "_RoofMatl_Roll",
    "_Condition2_RRAe", "_Condition2_RRAn", "_Condition2_RRNn",
    "_Heating_Floor", "_Heating_OthW",
    "_Electrical_Mix",
    "_MiscFeature_TenC",
    "_GarageQual_Ex", "_PoolQC_Fa"
]
combined_train.drop(drop_columns, axis=1, inplace=True)

refactored_data_frame = convert_and_combine(test)
refactored_data_frame = refactor(refactored_data_frame, neighborhood_bin_test ,"NeighborhoodBin", None)
combined_test = combined_test.join(refactored_data_frame)

#is not available in train
combined_test.drop(["_MSSubClass_150"], axis=1, inplace=True)

#drop not helpful columns
drop_columns = [
    "_Condition2_PosN",  # only two are not zero
    "_MSZoning_C (all)",
    "_MSSubClass_160",
]
combined_train.drop(drop_columns, axis = 1 , inplace = True)
combined_test.drop(drop_columns, axis = 1 , inplace = True)

label = pd.DataFrame(index=combined_train.index ,columns=["SalePrice"])
label["SalePrice"] = np.log(train["SalePrice"])

print("Training set size:", combined_train.shape)
print("Test set size:", combined_test.shape)

#lasso
y_prediction_lasso = lasso(combined_train,combined_test,label)

prediction = pd.DataFrame(y_prediction_lasso , index=test["Id"] ,columns=["SalePrice"])
prediction.to_csv('output_lasso.csv', header=True, index_label='Id')
prediction.to_csv('P2_submission.csv', header=True, index_label='Id')

#gradient boosting regression
y_prediction_gb = gradient_boosting(combined_train ,combined_test,label)
prediction = pd.DataFrame(y_prediction_gb , index=test["Id"] ,columns=["SalePrice"])
prediction.to_csv('output_gb.csv', header=True, index_label='Id')

#random forest
y_prediction_rf = random_forest(combined_train,combined_test,label)
prediction = pd.DataFrame(y_prediction_rf , index=test["Id"] ,columns=["SalePrice"])
prediction.to_csv('output_rf.csv', header=True, index_label='Id')
