from flaskapp import db,login_manager
from flask_login import UserMixin

class User(db.Model,UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    #image_file=db.Column(db.String(50),nullable=False,default='default.jpg')
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    diabete_history=db.Column(db.String(500))
    heart_result=db.Column(db.String(200))
    kidney=db.Column(db.String(200))
    liver=db.Column(db.String(200))
    lungs=db.Column(db.String(200))

    


class diabete(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    processedmeat=db.Column(db.Integer)
    fired_food=db.Column(db.Integer)
    soft_drink=db.Column(db.Integer)
    white_rice=db.Column(db.Integer)
    physical_excerise=db.Column(db.Integer)
    obesity=db.Column(db.Integer)
    family_history=db.Column(db.Integer)
    father_age=db.Column(db.Integer)
    age=db.Column(db.Integer)
    blood_pressure=db.Column(db.Integer)
    excessive_stress=db.Column(db.Integer)
    smoking=db.Column(db.Integer)
    alcoholic=db.Column(db.Integer)
    sleep_problem=db.Column(db.Integer)
    result=db.Column(db.Integer)
    



class heart(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    Age=db.Column(db.Integer)
    Gender=db.Column(db.Integer)
    family_history=db.Column(db.Integer)
    blood_pressure=db.Column(db.Integer)
    HyperTension=db.Column(db.Integer)
    smoking=db.Column(db.Integer)
    stress=db.Column(db.Integer)
    alcoholic=db.Column(db.Integer)
    BodyWeight=db.Column(db.Integer)
    Excessive_intakeof_salt=db.Column(db.Integer)
    Excessive_intakeof_coffee=db.Column(db.Integer)
    result=db.Column(db.Integer)

class kidney(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        Age=db.Column(db.Integer)
        FamilyHistory=db.Column(db.Integer)
        PhysicalExcerise=db.Column(db.Integer)
        obesity=db.Column(db.Integer)
        Hypertension=db.Column(db.Integer)
        HeartDieases=db.Column(db.Integer)
        Smoking=db.Column(db.Integer)
        excessive_painkillers=db.Column(db.Integer)
        Alcohol=db.Column(db.Integer)
        diabetes=db.Column(db.Integer)
        result=db.Column(db.Integer)

class liver(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        Gender=db.Column(db.Integer)
        Family_History=db.Column(db.Integer)
        Alcohol=db.Column(db.Integer)
        smoking=db.Column(db.Integer)
        Sleep_disorder=db.Column(db.Integer)
        obesity=db.Column(db.Integer)
        excessive_medication=db.Column(db.Integer)
        excessive_painkillers=db.Column(db.Integer)
        diabetes=db.Column(db.Integer)
        result=db.Column(db.Integer)

class lungs(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        Age=db.Column(db.Integer)
        family_history=db.Column(db.Integer)
        smoking=db.Column(db.Integer)
        allergies=db.Column(db.Integer)
        weight=db.Column(db.Integer)
        Exercise=db.Column(db.Integer)
        Location=db.Column(db.Integer)
        Result=db.Column(db.Integer)


    
