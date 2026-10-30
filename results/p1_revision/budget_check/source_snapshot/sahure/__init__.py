"""Self-supervised adaptive Hankel estimators; isolated from existing studies."""
__all__ = ['AdaptiveHankelEstimator','ModelConfig']


def __getattr__(name):
    # Let the CLI configure numerical-library threads before importing torch.
    if name in __all__:
        from .model import AdaptiveHankelEstimator,ModelConfig
        return {'AdaptiveHankelEstimator':AdaptiveHankelEstimator,'ModelConfig':ModelConfig}[name]
    raise AttributeError(name)
