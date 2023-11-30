from models.activity_models import ActivityModelA

class ActivityBenchmark:

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.result = None

    def run(self, data):
        if self.result is None:
            self.result = self.model.calculate(data)
        return self.result

    def get_ranking(self):
        if self.result is None:
            self.run()
        return self.result

