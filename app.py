from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

with open("random_forest.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET" , "POST"])
def predict():
    if(request.method=='POST'):
        Pclass=int(request.form.get("pclass"))
        Sex=int(request.form.get("Sex"))
        Age=int(request.form.get("Age"))
        SibSp=int(request.form.get("SibSp"))
        Parch=int(request.form.get("Parch"))
        Fare=int(request.form.get("Fare"))
        Deck=int(request.form.get("Deck"))
        Embarked=int(request.form.get("Embarked"))
        Title=int(request.form.get("Title"))
        Age_Class=Age*Pclass
        Relatives=SibSp+Parch
        Fare_Per_Person=Fare/(Relatives+1)
        not_alone=1
        if(Relatives):
            not_alone=0
        features = [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Relatives, not_alone, Deck, Title, Age_Class,  Fare_Per_Person] 
        
        prediction = model.predict([features])[0]
        result=0
        var=0
        if(prediction==1):
            result="There are 92% chances that you were get Survived."
            var=1
        else:
            result="There are 92% chances that you were not get Survived, but still there were 8% chances of survival."
        return render_template("index.html", result=result, var=var)

if __name__ == "__main__":
    app.run(debug=True)
