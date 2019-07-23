

class GA:

    def __init__(self, population_templates, population_size,
                 target_percentage, mutation_rate,
                 survival_threshold, initial_mutation_rate):

        self.population_templates = population_templates
        self.population_size = population_size
        self.target_percentage = target_percentage
        self.mutation_rate = mutation_rate
        self.survival_threshold = survival_threshold
        self.initial_mutation_rate = initial_mutation_rate

    def init(self):

        pass


