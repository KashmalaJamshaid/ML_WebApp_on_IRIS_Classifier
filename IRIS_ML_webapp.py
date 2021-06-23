# Importing all necessary libraries
import streamlit as st
import pickle

# We will load ML models into the respective pickle files. Here 'rb' shows the file is in the read mode
linear_regression_model = pickle.load(open('linear_regression_model.pkl', 'rb'))
logistic_regression_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))


# We don't want to display output as numbers i.e 0,1 or 2. Therefore, this class will display the name of the flowers
# instead of numbers.
def classify_data(num):
    if num < 0.5:
        return 'Setosa'
    elif num < 1.5:
        return 'Versicolor'
    else:
        return 'Virginica'


# Main function
def main():
    st.title("Machine Learning WebApp based on IRIS Classifier using Streamlit")

    # HTML code for webpage
    html_in_app = """
    <div style="background-color:#581845  ;padding:10px">
    <h2 style="color:white;text-align:center;">IRIS Classification using ML models</h2>
    </div>
    """
    # For executing html code in the application
    st.markdown(html_in_app, unsafe_allow_html=True)

    # For displaying models on sidebar
    classification_models = ['Linear Regression', 'Logistic Regression', 'SVM']
    option = st.sidebar.selectbox('Which model would you like to use?', classification_models)

    # Here, subheader will display ML model options
    st.subheader(option)

    # Slider for taking sepal length, width and petal length, width from the user. Here, min value is 0 & max value
    # is 10.
    sepal_length = st.slider('Select Sepal Length', 0.0, 10.0)
    sepal_width = st.slider('Select Sepal Width', 0.0, 10.0)
    petal_length = st.slider('Select Petal Length', 0.0, 10.0)
    petal_width = st.slider('Select Petal Width', 0.0, 10.0)

    # Creating 2D array, since ML model is trained on 2D array.
    inputs = [[sepal_length, sepal_width, petal_length, petal_width]]

    # This statement will run once classify button is clicked. It will call different ML models.
    if st.button('Classify'):
        if option == 'Linear Regression':
            st.success(classify_data(linear_regression_model.predict(inputs)))
        elif option == 'Logistic Regression':
            st.success(classify_data(logistic_regression_model.predict(inputs)))
        else:
            st.success(classify_data(svm_model.predict(inputs)))


# Calling main function
if __name__ == '__main__':
    main()
