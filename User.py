class User:
    def __init__(self, username, forename, surname, age, gpaq, three_f, sports=[]):
        self.username = username
        self.forename = forename
        self.surname = surname
        self.age = age
        self.sports = sports
        self.gpaq = gpaq
        self.three_f = three_f
        self.steps = 0

    @staticmethod
    def from_dict(source):
        return User(username=source['username'], forename=source['forename'], surname=source['surname'], age=source['age'], gpaq=source['gpaq'], three_f=source['three_f'])
        
    
    def to_dict(self):
        return vars(self)