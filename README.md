# master-project-cleaned

Wind gusts—short-lived bursts of high-speed wind—pose substantial risks to infrastructure and communi-
ties across Europe. With wind gusts becoming more intense under changing climate conditions, accurately
predicting these extreme events is increasingly important. This thesis aims to identify and predict key at-
mospheric processes that precede strong wind gusts, focusing on two primary weather systems: convective
cells in summer and extra-tropical cyclones in winter. The pilot study analyzed summer convective cells
from 2011 to 2020, followed by a wintertime investigation of extra-tropical cyclones using a dataset of 96
historical European storms and ERA5 reanalysis spanning 1990 to 2021. First, we extract storm-centered
meteorological fields—such as temperature, humidity, and wind—from a moving 8° × 8° region around each
cyclone. We then employ spatial and temporal dimension-reduction techniques. Statistically summarizing
each field (mean, max, min, standard deviation) reduces the spatial dimensionality, while Principal Com-
ponent Analysis (PCA) captures dominant temporal modes, reducing the total number of features. Next,
k-means clustering groups Europe into 15 regions to ensure sufficient sample size in each cluster. We also
transform raw gust speeds into a GEV-based percentile metric, improving the comparability of European re-
gions. Linear regression in this reduced-dimensional space predicts wind gust intensity more effectively than
either a climatological baseline or non-linear models. Key predictors—baroclinic instability, moisture trans-
port, and near-surface wind intensification—commonly emerge 24–36 hours before landfall, demonstrating
that storm history is critical for anticipating damaging gusts. Case studies of two severe storms, Vivian
(1990) and Lothar (1999), validate the model’s ability to capture relevant physical processes. Overall, this
thesis shows that combining physically guided dimensionality reduction with simple regression models can
uncover meaningful gust precursors and outperform simpler baselines, offering a viable path for improved
windstorm forecasting despite limited historical data.