from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
from sklearn.pipeline import Pipeline

def build_baseline_pipelines(preprocessor):
    baselines = {
        'Ridge': Pipeline([('preproc', preprocessor), ('model', Ridge())]),
        'ElasticNet': Pipeline([('preproc', preprocessor), ('model', ElasticNet(max_iter=5000))]),
        'HGB': Pipeline([('preproc', preprocessor), ('model', HistGradientBoostingRegressor(random_state=42))])
    }
    return baselines

def build_stacking_pipeline(preprocessor, base_estimators=None, final_estimator=None):
    if base_estimators is None:
        base_estimators = [
            ('ridge', Ridge()),
            ('elastic', ElasticNet(max_iter=5000)),
            ('hgb', HistGradientBoostingRegressor(random_state=42, max_iter=200))
        ]
    if final_estimator is None:
        final_estimator = Ridge()
    stack = StackingRegressor(estimators=base_estimators, final_estimator=final_estimator, passthrough=False, n_jobs=-1)
    stack_pipe = Pipeline([('preproc', preprocessor), ('model', stack)])
    return stack_pipe
