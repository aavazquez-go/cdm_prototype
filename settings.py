class Models:

    def __init__(self, max_models):
        self.max_models = max_models
        self.current_count = 0
        self.models = []

    def add_model(self, model):
        if self.current_count < self.max_models:
            self.current_count += 1
            self.models.append(model)
            return True
        return False

    def reset(self):
        self.current_count = 0
        self.models = []

    def remaining(self):
        return self.max_models - self.current_count

    def is_full(self):
        return self.current_count >= self.max_models
    
    def get_models(self):
        return self.models
    
    def get_model(self, index):
        if 0 <= index < self.current_count:
            return self.models[index]
        return None
    
