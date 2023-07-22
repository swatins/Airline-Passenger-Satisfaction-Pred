import os,sys
from flask import Flask, render_template, request , jsonify 
from src.pipeline.prediction_pipeline import PredictionPipeline, AirlineClass
from src.logger import logging
from src.pipeline.batch_pred import batch_prediction
from src.components.data_transformation import DataTransformationConfig
from src.pipeline.transform_pipeline import Training_Pipeline
from werkzeug.utils import  secure_filename

UPLOAD_FOLDER = 'batch_prediction/Uploaded_CSV_FILE'
model_file_path ='artifacts/pkls/model.pkl'
transformer_file_path='artifacts/pkls/preprocessor.pkl'

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'csv'}

@app.route('/')
def home_page():
    logging.info("Main page is loaded !!")
    return render_template('Index.html')

#Single Prediction Option

@app.route("/Single_pred", methods= ["GET","POST"])

def predict_data():
    if request.method == "GET":
        logging.info("Single Prediction page is loaded !!")
        return render_template("Single_pred.html")
    
    else:
        data = AirlineClass(
                Gender=str(request.form.get("Gender")),
                Customer_Type=str(request.form.get("Customer_Type")),#loyal / disloyal
                Age =int(request.form.get("Age")),
                Type_of_Travel=str(request.form.get("Type_of_Travel")), #personal/business
                Class=str(request.form.get("Class")), #business/eco/eco plus
                Flight_Distance=int(request.form.get("Flight_Distance")),
                Inflight_wifi_service= int(request.form.get("Inflight_wifi_service")),
                Departure_Arrival_time=int(request.form.get("Departure_Arrival_time")), 
                Ease_of_Online_booking=int(request.form.get("Ease_of_Online_booking")),
                Gate_location=int(request.form.get("Gate_location")), 
                Food_and_drink=int(request.form.get("Food_and_drink")),
                Online_boarding=int(request.form.get("Online_boarding")),
                Seat_comfort=int(request.form.get("Seat_comfort")),
                Inflight_entertainment=int(request.form.get("Inflight_entertainment")),
                On_board_service=int(request.form.get("On_board_service")),
                Leg_room_service=int(request.form.get("Leg_room_service")),
                Baggage_handling=int(request.form.get("Baggage_handling")),
                Checkin_service=int(request.form.get("Checkin_service")),
                Inflight_service=int(request.form.get("Inflight_service")),
                Cleanliness=int(request.form.get("Cleanliness")),
                Departure_Delay_in_Minutes=int(request.form.get("Departure_Delay_in_Minutes")),
                #Arrival_Delay_in_Minutes=int(request.form.get("Arrival_Delay_in_Minutes"))
            
            )
        
    final_data= data.get_data_into_DataFrame()  
    logging.info("Sending data to pred pipeline !!")
    Pred_Pipeline = PredictionPipeline()
    pred= Pred_Pipeline.predict(final_data)
    
    result = pred 
    logging.info("Got the result  !!")
    if result == 0:
        logging.info(f"The Passenger is {result} with the airline: !!")
        return render_template("result.html",final_result= "The Passenger is not satisfied with the airline:{}".format(result))
    
    elif result == 1:
        logging.info(f"The Passenger is {result} with the airline: !!")
        return render_template("result.html",final_result= "The Passenger is satisfied with the airline:{}".format(result))
        
    
#Batch Prediction Option

@app.route("/batch_pred", methods=['GET','POST'])
def perform_batch_prediction():
    
    
    if request.method == 'GET':
        return render_template('batch_pred.html')
    else:
        file = request.files['csv_file']  # Update the key to 'csv_file'
        # Directory path
        directory_path = UPLOAD_FOLDER
        # Create the directory
        os.makedirs(directory_path, exist_ok=True)

 # Check if the file has a valid extension
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
            # Delete all files in the file path
            for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Save the new file to the uploads directory
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print(file_path)

            logging.info("CSV received and Uploaded")    

            # Perform batch prediction using the uploaded file
            batch = batch_prediction(file_path,
                                    model_file_path,
                                    transformer_file_path
                                    )
            pred_path=batch.start_batch_prediction()
            new_line = '\n'
            output = f"Batch Prediction Done!!!. '{new_line}' Please find the predictions.csv at : '{new_line}'    '{pred_path}'"
            return render_template("result.html", final_result=output, prediction_type='batch')
        else:
            return render_template('result.html', prediction_type='batch', error='Invalid file type')
        

# Train Model Option


@app.route('/train', methods=['GET', 'POST'])
def train():
    #if request.method == 'GET':
     #   return render_template('result.html')
   # else:
        try:
            pipeline = Training_Pipeline()
            #pipeline.main()
            print ("Training Completed!!")
            output = f"Training pipeline completed!!."
            return render_template("result.html", final_result=output, prediction_type='train')
            

        except Exception as e:
            logging.error(f"{e}")
            error_message = str(e)
            return render_template("result.html", final_result=error_message, prediction_type='train')
           # return render_template('Index.html', error=error_message)
        

if __name__ == "__main__":
     app.run( debug=True)
     
     # host = "0.0.0.0.",
