class FunctionRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(func):
            self._registry[name] = func
            return func
        return decorator

    def __getattr__(self, name):
        if name in self._registry:
            return self._registry[name]
        raise AttributeError(f"No function registered with name '{name}'")

registry = FunctionRegistry()
