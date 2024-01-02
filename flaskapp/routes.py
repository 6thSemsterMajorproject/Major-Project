from flask import Flask, redirect,render_template,request,flash,url_for,session
from flaskapp import app,db,login_manager
from flaskapp.models import User,diabete,heart,Kidney,lungs
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user,logout_user
import pandas as pd
import numpy as np
import joblib
import bcrypt
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime
import random
from io import BytesIO
import base64
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

def current_data():
    cur_date=datetime.now()
    for_date=cur_date.strftime("%y/%m/%d")
    return for_date
def List_validator(my_list):
    flag=0
    for i in my_list:
        if i<0:
            flag=1
    return flag
def  getting_the_data(s):
    result=str(s)
    user=User.query.filter_by(username=current_user.username).first()
    if  "Diabetes" in result:
        data=user.diabete_history
    elif "Heart_Diesease" in result:
        data=user.heart_result
    elif "Kidney_Disease" in result:
        data=user.kidney
    elif"Liver_Disease" in result:
        data=user.liver
    elif "Asthama" in result:
        data=user.Asthama
    data=str(data).replace("None","")
    return data
#class Section
class account_data:
    def data_spearator(self,data):
        data=str(data)
        parts = data.split(':')[1:]  # Split from the first ':' to the end
        result1 = [part.split(';')[0] for part in parts]
        self.result=[]
        self.date=[]
        self.percentage=[]
        for i,item in enumerate(result1):
            if i%3==0:
                self.result.append(item)
            elif i%3==1:
                self.date.append(item)
            else:
                self.percentage.append(item)

#routes section 
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.errorhandler(Exception)
def handle_error(error):
    exception_name = error.__class__.__name__
    return "This is Error handler"+str(error)
@app.route("/")
def index():
    return render_template("index.html")
@app.route("/Logout")
@login_required
def Logout():
    logout_user()
    return render_template("index.html")
@app.route("/Account")
@login_required
def Account():
    try:
        user=User.query.filter_by(username=current_user.username).first()
        obj=account_data()
        obj.data_spearator(user.diabete_history)
    except Exception as e:
        print(e)    
    return  render_template("account.html",username=user.username,email=user.email,dia=obj)#,graph_path=graph_path)
@app.route("/account_heart")
@login_required
def Account_heart():
    user=User.query.filter_by(username=current_user.username).first()
    obj=account_data()
    obj.data_spearator(user.heart_result)
    return  render_template("account_heart.html",username=user.username,email=user.email,her=obj)
@app.route("/account_kidney")
@login_required
def Account_Kidney():
    user=User.query.filter_by(username=current_user.username).first()
    obj=account_data()
    obj.data_spearator(user.kidney)
    return  render_template("account_kidney.html",username=user.username,email=user.email,dia=obj)
@app.route("/account_liver")
@login_required
def Account_Liver():
    user=User.query.filter_by(username=current_user.username).first()
    obj=account_data()
    obj.data_spearator(user.liver)
    return  render_template("account_liver.html",username=user.username,email=user.email,dia=obj)
@app.route("/account_lungs")
@login_required
def Account_Lungs():
    user=User.query.filter_by(username=current_user.username).first()
    obj=account_data()
    obj.data_spearator(user.Asthama)
    return  render_template("account_lungs.html",username=user.username,email=user.email,dia=obj)
@app.route("/AboutUs")
def AboutUs():
    return render_template("about.html")
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
    user=User(username=Username,email=Email,password=Password)
    try:
        db.session.add(user)
        db.session.commit()
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
    try:
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
        flag=int(List_validator(diabetes_details) )
        if flag==1:
            return "You Have Given an Illegal Value Please Retry"
            
        else:
            data = get_diabete_data()
            if data:
            # Extracting features and target values from data
                X = [[item.processedmeat, item.fired_food, item.soft_drink, item.white_rice, item.physical_excerise, 
                        item.obesity, item.family_history, item.father_age, item.age, item.blood_pressure, 
                        item.excessive_stress, item.smoking, item.alcoholic, item.sleep_problem] for item in data]
                
                y = [item.result for item in data]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # The Testing is 20% and training is 80% 
            # Create and train a Random Forest classifier model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
                # Save the trained model
            joblib.dump(model, 'diabetes_risk_model.pkl')
                # Load the model
            loaded_model = joblib.load('diabetes_risk_model.pkl')
            input_data = [diabetes_details]
            prediction = loaded_model.predict_proba(input_data)
            y_pred = loaded_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            user=User.query.filter_by(username=current_user.username).first()
            data=getting_the_data("Diabetes")
            data=str(data)
            if prediction[0][1] >= 0.5:    
                user.diabete_history=data+"Diabetes:Yes;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction[0][1]*100)+";"
                db.session.commit()
                return  render_template("result.html",result="Diabetes:Yes;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction[0][1]*100)+";")
            else:
                user.diabete_history=data+"Diabetes:No;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction[0][1]*100)+";"
                db.session.commit()
                return  render_template("result.html",result="Diabetes:No;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction[0][1]*100)+";")
    except Exception as e:
        print(e)
    # Heart Module 
@app.route("/heart")
def Heart():
    return render_template("heart.html")
# Heart Diesease Predictor Uses LogisticRegression
@app.route("/heartsub",methods=["GET","POST"])
def Heartsub():
    try:    
        Heart_details=[]
        Heart_details.append(int(request.form.get("age")))
        Heart_details.append(request.form.get("checkbox2"))
        Heart_details.append(request.form.get("checkbox7"))
        Heart_details.append(request.form.get("checkbox5"))
        Heart_details.append(request.form.get("checkbox10"))
        Heart_details.append(request.form.get("checkbox9"))
        Heart_details.append(request.form.get("checkbox11"))
        Heart_details.append(int(request.form.get("Weight")))
        Heart_details.append(request.form.get("checkbox12"))
        Heart_details.append(request.form.get("checkbox3"))
        Heart_details=List_replacer(Heart_details)    
        flag=int(List_validator(Heart_details))
        if flag==1:
            return '''
            <script>alert('You Have Entered  invalid data');</script>
                    '''
        else:
            query="select Age,Gender,family_history,HyperTension,smoking,stress,alcoholic,Bodyweight,Excessive_intakeof_salt,Excessive_intakeof_coffee,result from heart"
            conn = sqlite3.connect("C:\\AIhealthpro\\instance\\AIhealthpro.db")
            data = pd.read_sql_query(query, conn)
            # data=db.engine.execute(query)
            X = data.drop('result',axis=1)  # Features
            y = data['result']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # The Testing is 20% and training is 80% 
            model = LogisticRegression(random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, 'Heart_risk_model.pkl')
            loaded_model = joblib.load('Heart_risk_model.pkl')
            input_details=[Heart_details]
            prediction = loaded_model.predict(input_details)
            user=User.query.filter_by(username=current_user.username).first()
            data=getting_the_data("Heart_Diesease")
            data=str(data)
            if prediction==1:
                try:
                    user.heart_result=data+"Heart_Diesease:Yes;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction*100)+";"
                    db.session.commit()
                except Exception as e:
                    print("this is Exception",e)
                return  render_template("result.html",result="Heart_Diesease:Yes;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction*100)+";")
            else:
                user.heart_result=data+"Heart_Diesease:No;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction*100)+";"
                db.session.commit()
                return  render_template("result.html",result="Heart_Diesease:No;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction*100)+";")
    except Exception as e:
        return str(e)
@app.route("/Kidney")
def Kidney():
    return render_template("Kidney.html")
@app.route("/kidneysub",methods=["GET","POST"])
def Kidneysub():
    try:
        kidney_details=[]
        kidney_details.append(int(request.form.get("age")))
        kidney_details.append(request.form.get("checkbox1"))
        kidney_details.append(request.form.get("checkbox3"))
        kidney_details.append(request.form.get("checkbox4"))
        kidney_details.append(request.form.get("checkbox2"))
        kidney_details.append(request.form.get("checkbox5"))
        kidney_details.append(request.form.get("checkbox6"))
        kidney_details.append(request.form.get("checkbox7"))
        kidney_details.append(request.form.get("checkbox8"))
        kidney_details.append(request.form.get("checkbox9"))
        kidney_details=List_replacer(kidney_details)        
        print(kidney_details)
        flag=int(List_validator(kidney_details))
        if flag==1:
            return "You Have entered illegal argument"
        else:
            conn=sqlite3.connect("C:\\AIhealthpro\\instance\\AIhealthpro.db")
            cursor = conn.cursor()
            cursor.execute('SELECT Age,family_history,physical_excerise,obesity,Hypertension,HeartDieases,smoking,painkiller,alcoholic,diabetes,result FROM kidney')
            data=cursor.fetchall()
            conn.close()
            data=pd.DataFrame(data)
            X=data.iloc[:,:-1] 
            y=data.iloc[:,-1] 
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train,y_train)
            joblib.dump(model,"Kidney_risk_model.pkl")
            loaded_model=joblib.load("Kidney_risk_model.pkl")
            try:
                input_data=[kidney_details]
                prediction=loaded_model.predict(input_data)
            except Exception as e:
                print(e)
            user=User.query.filter_by(username=current_user.username).first()
            data=getting_the_data("Kidney_Disease")
            data=str(data)
            if prediction>=0.5:
                user.kidney=data+"Kidney_Disease:Yes;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction*100)+";"
                db.session.commit()
                return  render_template("result.html",result="Kidney_Disease:Yes;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction*100)+";")
            else:
                user.kidney="Kidney_Disease:No;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction*100)+";"
                db.session.commit()
                return  render_template("result.html",result="Kidney_Disease:No;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction*100)+";")
    except Exception as e:
        print(e)
        return str(e)
@app.route("/Liver")
def Liver():
    return render_template("Liver.html")
@app.route("/Liversub",methods=["GET","POST"])
def Liversub():
    Liver_detials=[]
    Liver_detials.append(int(request.form.get('age')))
    Liver_detials.append(request.form.get("checkbox1"))
    Liver_detials.append(request.form.get("checkbox3"))
    Liver_detials.append(request.form.get("checkbox4"))
    Liver_detials.append(request.form.get("checkbox2"))
    Liver_detials.append(request.form.get("checkbox5"))
    Liver_detials.append(request.form.get("checkbox6"))
    Liver_detials.append(request.form.get("checkbox7"))
    Liver_detials.append(request.form.get("checkbox8"))
    Liver_detials=List_replacer(Liver_detials)
    print(Liver_detials)
    flag=int(List_validator(Liver_detials))
    print(flag)
    if flag ==1:
        return "You Have entered illegal argument"
    else:
        data=pd.read_csv("C:\\AIhealthpro\\flaskapp\\my_csv_file\\liver.csv")
        X=data.drop('result',axis=1)
        y=data['result']
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # The Testing is 20% and training is 80% 
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred=model.predict(X_test)
            accuracy=accuracy_score(y_test,y_pred)
            joblib.dump(model, 'Liver_risk_model.pkl')
            loaded_model = joblib.load('Liver_risk_model.pkl')
            prediction=loaded_model.predict([Liver_detials])
        except Exception as e:
            print("This is Exception",e)
        user=User.query.filter_by(username=current_user.username).first()
        predict=int(prediction)
        data=getting_the_data("Liver_Disease")
        data=str(data)
        if predict==0:
            user.liver=data+"Liver_Disease:Yes;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(predict*100)+";"
            db.session.commit()
            return  render_template("result.html",result="Liver_Disease:Yes;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(predict*100)+";")
        else:
            user.liver=data+"Liver_Disease:NO;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(predict*100)+";"
            db.session.commit()
            return  render_template("result.html",result="Liver_Disease:NO;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(predict*100)+";")
@app.route('/Lungs')
def Lungs():
    return render_template('lungs.html')
@app.route('/lungssub',methods=['GET','POST'])
def lungssub():
        try:
            Lungs_details=[]
            Lungs_details.append(int(request.form.get("age")))
            Lungs_details.append(request.form.get("checkbox1"))
            Lungs_details.append(request.form.get("checkbox2"))
            Lungs_details.append(request.form.get("checkbox3"))
            Lungs_details.append(int(request.form.get("Weight")))
            Lungs_details.append(request.form.get("checkbox4"))
            Lungs_details.append(request.form.get("checkbox5"))
            Lungs_details=List_replacer(Lungs_details)
            flag=int(List_validator(Lungs_details))
            if flag:
                return "You have entered illegal argument"
            data=pd.read_csv("C:\\AIhealthpro\\flaskapp\\my_csv_file\\lungs.csv")
            X=data[['Age','family_history','smoking','allergies','weight','Exercise','Location']]
            y=data['Result']
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
            model=LinearRegression()
            model.fit(X_train,y_train)
            joblib.dump(model, 'Lungs_risk_model.pkl')
            loaded_model = joblib.load('Lungs_risk_model.pkl')
            input_details=[Lungs_details]
            prediction = loaded_model.predict(input_details)
            user=User.query.filter_by(username=current_user.username).first()
            data=getting_the_data("Asthama")
            data=str(data)
            if prediction >= 0.5:
                user.lungs=data+"Asthama:Yes;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction*100)+";"
                return  render_template("result.html",result="Asthama:Yes;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction*100)+";")
            else:
                user.lungs=data+"Asthama:No;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction*100)+":"
                return  render_template("result.html",result="Asthama:No;\n"+"Time:"+str(current_data())+";\n"+"stage:"+str(prediction*100)+":")
        except Exception as e:
            print(e)
