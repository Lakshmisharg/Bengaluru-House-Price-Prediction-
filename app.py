import pandas as pd
import numpy as np
import gradio as gr
import sklearn
import pickle


x = pd.read_csv('x_dataframe.csv') # x is independent variable of dataset, this line is required because of x was not defined in this script.

def ml_model(Model,location,total_sqft,bath,bhk):
    loc_index = np.where(x.columns==location)[0][0]

    X = np.zeros(len(x.columns))
    X[0] = total_sqft
    X[1] = bath
    X[2] = bhk
    if loc_index >= 0:
        X[loc_index] = 1

    
    
    df=[X]
    

    model = pickle.load(open(r'dt_model.pkl','rb'))          # decision tree
    model1 = pickle.load(open(r'lr_model.pkl','rb'))     #linear regression
    
    # prediction using decision tree
    if Model == 'Decision Tree':

        pred = model.predict(df)
        price = str(int(pred[0]*100000))
        output_text = 'price for required apartment would be approximately ' + price + ' Rupees'
        return output_text
    
    # prediction using linear regression
    elif Model == 'Linear Regression':
        
        pred1 = model1.predict(df)
        price = str(int(pred1[0]*100000))
        output_text = 'price for required apartment would be approximately ' + price + ' Rupees'
        return output_text
        
interface = gr.Interface(
    title="Bengaluru House Price Prediction",
    fn=ml_model,
    inputs=[
        gr.Dropdown(['Linear Regression','Decision Tree'], label="Select a Model"),
        gr.Dropdown(['Electronic City Phase II', 'Chikka Tirupathi', 'Uttarahalli',
       'Lingadheeranahalli', 'Kothanur', 'Whitefield', 'Old Airport Road',
       'Rajaji Nagar', 'Marathahalli', '7th Phase JP Nagar',
       'Gottigere', 'Sarjapur', 'Mysore Road', 'Bisuvanahalli',
       'Raja Rajeshwari Nagar', 'Kengeri', 'Binny Pete', 'Thanisandra',
       'Bellandur', 'Electronic City', 'Ramagondanahalli', 'Yelahanka',
       'Hebbal', 'Kasturi Nagar', 'Kanakpura Road',
       'Electronics City Phase 1', 'Kundalahalli', 'Chikkalasandra',
       'Murugeshpalya', 'Sarjapur  Road', 'HSR Layout', 'Doddathoguru',
       'KR Puram', 'Bhoganhalli', 'Lakshminarayana Pura', 'Begur Road',
       'Varthur', 'Bommanahalli', 'Gunjur', 'Devarachikkanahalli',
       'Hegde Nagar', 'Haralur Road', 'Hennur Road', 'Kothannur',
       'Kalena Agrahara', 'Kaval Byrasandra', 'ISRO Layout',
       'Garudachar Palya', 'EPIP Zone', 'Dasanapura', 'Kasavanhalli',
       'Sanjay nagar', 'Domlur', 'Sarjapura - Attibele Road',
       'Yeshwanthpur', 'Chandapura', 'Nagarbhavi', 'Devanahalli',
       'Ramamurthy Nagar', 'Malleshwaram', 'Akshaya Nagar', 'Shampura',
       'Kadugodi', 'LB Shastri Nagar', 'Hormavu', 'Vishwapriya Layout',
       'Kudlu Gate', '8th Phase JP Nagar', 'Bommasandra Industrial Area',
       'Anandapura', 'Vishveshwarya Layout', 'Kengeri Satellite Town',
       'Kannamangala', 'Hulimavu', 'Mahalakshmi Layout', 'Hosa Road',
       'Attibele', 'CV Raman Nagar', 'Kumaraswami Layout', 'Nagavara',
       'Hebbal Kempapura', 'Vijayanagar', 'Pattandur Agrahara',
       'Nagasandra', 'Kogilu', 'Panathur', 'Padmanabhanagar',
       '1st Block Jayanagar', 'Kammasandra', 'Dasarahalli', 'Magadi Road',
       'Koramangala', 'Dommasandra', 'Budigere', 'Kalyan nagar',
       'OMBR Layout', 'Horamavu Agara', 'Ambedkar Nagar',
       'Talaghattapura', 'Balagere', 'Jigani', 'Gollarapalya Hosahalli',
       'Old Madras Road', 'Kaggadasapura', '9th Phase JP Nagar', 'Jakkur',
       'TC Palaya', 'Giri Nagar', 'Singasandra', 'AECS Layout',
       'Mallasandra', 'Begur', 'JP Nagar', 'Malleshpalya', 'Munnekollal',
       'Kaggalipura', '6th Phase JP Nagar', 'Ulsoor', 'Thigalarapalya',
       'Somasundara Palya', 'Basaveshwara Nagar', 'Bommasandra',
       'Ardendale', 'Harlur', 'Kodihalli', 'Narayanapura',
       'Bannerghatta Road', 'Hennur', '5th Phase JP Nagar', 'Kodigehaali',
       'Billekahalli', 'Jalahalli', 'Mahadevpura', 'Anekal', 'Sompura',
       'Dodda Nekkundi', 'Hosur Road', 'Battarahalli', 'Sultan Palaya',
       'Ambalipura', 'Hoodi', 'Brookefield', 'Yelenahalli', 'Vittasandra',
       '2nd Stage Nagarbhavi', 'Vidyaranyapura', 'Amruthahalli',
       'Kodigehalli', 'Subramanyapura', 'Basavangudi', 'Kenchenahalli',
       'Banjara Layout', 'Kereguddadahalli', 'Kambipura',
       'Banashankari Stage III', 'Sector 7 HSR Layout', 'Rajiv Nagar',
       'Arekere', 'Mico Layout', 'Kammanahalli', 'Banashankari',
       'Chikkabanavar', 'HRBR Layout', 'Nehru Nagar', 'Kanakapura',
       'Konanakunte', 'Margondanahalli', 'R.T. Nagar', 'Tumkur Road',
       'Vasanthapura', 'GM Palaya', 'Jalahalli East', 'Hosakerehalli',
       'Indira Nagar', 'Kodichikkanahalli', 'Varthur Road', 'Anjanapura',
       'Abbigere', 'Tindlu', 'Gubbalala', 'Parappana Agrahara',
       'Cunningham Road', 'Kudlu', 'Banashankari Stage VI', 'Cox Town',
       'Kathriguppe', 'HBR Layout', 'Yelahanka New Town',
       'Sahakara Nagar', 'Rachenahalli', 'Yelachenahalli',
       'Green Glen Layout', 'Thubarahalli', 'Horamavu Banaswadi',
       '1st Phase JP Nagar', 'NGR Layout', 'Seegehalli', 'BEML Layout',
       'NRI Layout', 'ITPL', 'Babusapalaya', 'Iblur Village',
       'Ananth Nagar', 'Channasandra', 'Choodasandra', 'Kaikondrahalli',
       'Neeladri Nagar', 'Frazer Town', 'Cooke Town', 'Doddakallasandra',
       'Chamrajpet', 'Rayasandra', '5th Block Hbr Layout', 'Pai Layout',
       'Banashankari Stage V', 'Sonnenahalli', 'Benson Town',
       '2nd Phase Judicial Layout', 'Poorna Pragna Layout',
       'Judicial Layout', 'Banashankari Stage II', 'Karuna Nagar',
       'Bannerghatta', 'Marsur', 'Bommenahalli', 'Laggere',
       'Prithvi Layout', 'Banaswadi', 'Sector 2 HSR Layout',
       'Shivaji Nagar', 'Badavala Nagar', 'Nagavarapalya', 'BTM Layout',
       'BTM 2nd Stage', 'Hoskote', 'Doddaballapur', 'Sarakki Nagar',
       'Bharathi Nagar', 'HAL 2nd Stage', 'Kadubeesanahalli'], label="Select a location"),
        gr.Number(minimum=0,maximum=4500, label="total area in sqft."),
        gr.Dropdown([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], label="no of bathroom"),
        gr.Dropdown([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], label="no of bhk"),
    ],
    outputs='text',
    description="For better output, give input which makes sense real-estate wise. <br> like - 1 bhk apartment generally have minimum area of 300 sqft."
)

interface.launch()