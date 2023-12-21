from flask import Flask, redirect,render_template,request,flash,url_for,session
from flaskapp import app,db,login_manager
from flaskapp.models import User,diabete,heart
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user,logout_user

import pandas as pd
import numpy as np
import joblib
import bcrypt
 # function Section 
def hash_password(password):
    # Generate a salt and hash the password
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password

def check_password(password, hashed_password):
    # Check if the entered password matches the hashed password
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

def List_replacer(my_list):
    for i in range(len(my_list)):
        if my_list[i] is None:
            my_list[i] = 0
        if my_list[i] == "on":
            my_list[i]=1
    return my_list

def get_diabete_data():
    try:
        data_list=diabete.query.all()
    except Exception as e:
        print(e)
    return data_list
def get_heart_data():
    try:
        data=heart.query.all()
    except Exception as e:
        print(e)
    return data



#routes section 
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.errorhandler(Exception)
def handle_error(error):
    exception_name = error.__class__.__name__
    return'''
<script>
alert({err}+'Please Try Again')
</script>
'''.format(err=error)
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/Loader")
def loader():
    return render_template("Loading.html")
@app.route("/Logout")
@login_required
def Logout():
    logout_user()
    return render_template("index.html")
@app.route("/Account")
@login_required
def Account():
    print('hello')
    try:

   
        user=User.query.filter_by(username=current_user.username).first()
        print(user.username)
        print(user.email)
        print(user.diabete_history)
    except Exception as e:
        print(e)


    
        
    return  render_template("account.html",username=user.username,email=user.email,result=user.diabete_history)

@app.route('/AboutUs')
def about():
    return render_template('about.html')
@app.route("/homepage")
def homepage():
    return render_template("home.html")
@app.route("/loginpage")
def loginpage():
    return render_template("sign_page.html")
@app.route("/loginauth" ,methods=['GET','POST'])
def loginauth(): 
    Username=request.form.get("Uname")
    Password=request.form.get("pword")
    print("Hashing is done")
    try:
        specific_user_name = User.query.filter_by(username=Username).first()
        specific_user_password=check_password(Password,specific_user_name.password)
    except Exception as e:
        print(e)
    if specific_user_name and specific_user_password:
        login_user(specific_user_name)
        return render_template("home.html")
    else:
        return "Your Are Not A User"
@app.route("/signupauth",methods=['GET','POST'])
def signupauth():
    Username=request.form.get("name1")
    Email=request.form.get("email")
    Password=request.form.get("password")
    Password=hash_password(Password)
    print("Hasing is done ")
    user=User(username=Username,email=Email,password=Password)
    try:
        db.session.add(user)
        print("The Data Has Been Added")
        db.session.commit()
        print("The Data Has Been Added")
        
    except  Exception  as e:
        print(e)
    finally:
        return render_template("sign_page.html")
@app.route("/disinx",methods=["GET","POST"])
def disindx():
    return render_template("disindex.html")
@app.route("/diabetes",methods=["GET","POST"])
def diabetes():
    return render_template("diabetes.html")
#diabetes Precition is done by RandomForestClassifier
@app.route("/diabetessub",methods=["GET","POST"])
def dibetessub():
    diabetes_details=[]
    diabetes_details.append((request.form.get("checkbox")))
    diabetes_details.append((request.form.get("checkbox2")))
    diabetes_details.append((request.form.get("checkbox3")))
    diabetes_details.append((request.form.get("checkbox4")))
    diabetes_details.append((request.form.get("checkbox5")))
    diabetes_details.append((request.form.get("checkbox6")))
    diabetes_details.append((request.form.get("checkbox7")))
    diabetes_details.append(int(request.form.get("father_age")))
    diabetes_details.append(int(request.form.get("age")))
    diabetes_details.append((request.form.get("checkbox8")))
    diabetes_details.append((request.form.get("checkbox9")))
    diabetes_details.append((request.form.get("checkbox10")))
    diabetes_details.append((request.form.get("checkbox11")))
    diabetes_details.append((request.form.get("checkbox12")))
    diabetes_details=List_replacer(diabetes_details)

    print("The Data is getting ")
    data = get_diabete_data()
    print(data)
       
    if data:
    # Extracting features and target values from data
        X = [[item.processedmeat, item.fired_food, item.soft_drink, item.white_rice, item.physical_excerise, 
                item.obesity, item.family_history, item.father_age, item.age, item.blood_pressure, 
                item.excessive_stress, item.smoking, item.alcoholic, item.sleep_problem] for item in data]
        
        y = [item.result for item in data]
    print(X)
    print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # The Testing is 20% and training is 80% 
    
    # Create and train a Random Forest classifier model
    model = RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(X_train, y_train)

        # Save the trained model
    joblib.dump(model, 'diabetes_risk_model.pkl')

        # Load the model
    loaded_model = joblib.load('diabetes_risk_model.pkl')
    input_data = [diabetes_details]
    prediction = loaded_model.predict_proba(input_data)
    y_pred = loaded_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy*100)
    print(prediction[0][1])

    if prediction[0][1] >= 0.5:
        user=User.query.filter_by(username=current_user.username).first()
        user.diabete_history="Yes"
        print(user.diabete_history)
        db.session.commit()


        return '''
<script> alert("You Have High Risk Of Diabets")
</script>'''


    else:
        user=User.query.filter_by(username=current_user.username).first()
        user.diabete_history="No"
        print(user.diabete_history)
        db.session.commit()
        return '''
<script> alert("You Have low Risk Of Diabets")
</script>'''
    # Heart Module 
@app.route("/heart")
def Heart():
    return render_template("heart.html")
# Heart Diesease Predictor Uses LogisticRegression
@app.route("/heartsub",methods=["GET","POST"])
def Heartsub():
    try:    
        print("In The  Function ")
        Heart_details=[]
        Heart_details.append(int(request.form.get("age")))
        Heart_details.append(request.form.get("checkbox2"))
        Heart_details.append(request.form.get("checkbox7"))
        Heart_details.append(request.form.get("checkbox8"))
        Heart_details.append(request.form.get("checkbox5"))
        Heart_details.append(request.form.get("checkbox10"))
        Heart_details.append(request.form.get("checkbox9"))
        Heart_details.append(request.form.get("checkbox11"))
        Heart_details.append(int(request.form.get("Weight")))
        Heart_details.append(request.form.get("checkbox12"))
        Heart_details.append(request.form.get("checkbox3"))
        Heart_details=List_replacer(Heart_details)   
        print("getting data ")
        data=get_heart_data()
        print("got The data")
        if data:
        # Extracting features and target values from data
            X = [[item.Age, item.Gender, item.family_history, item.blood_pressure, item.HyperTension, 
                    item.smoking, item.stress, item.alcoholic, item.BodyWeight, item.Excessive_intakeof_salt, 
                    item.Excessive_intakeof_coffee] for item in data]
            
            y = [item.result for item in data]

        print(X)
        print(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # The Testing is 20% and training is 80% 
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)

            # Save the trained model
        joblib.dump(model, 'Heart_risk_model.pkl')

            
        loaded_model = joblib.load('Heart_risk_model.pkl')
        input_details=[Heart_details]
        prediction = loaded_model.predict_proba(input_details)
        print(prediction[0][1])


        if prediction[0][1] >= 0.5:
            return '''
                <script> alert("You Have High Risk Of Heart ")
                </script>'''
        else:
            return '''
                <script> alert("You Have low Risk Of Heart")
                </script>'''    
    except Exception as e:
        return str(e)
@app.route("/Kidney")
def Kidney():
    return render_template("Kidney.html")
@app.route("/kidneysub")
def Kidneysub():
    return "This is Kidney Sub"
@app.route("/Liver")
def Liver():
    return render_template("Liver.html")
