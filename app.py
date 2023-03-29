import streamlit as st
from streamlit_lottie import st_lottie
import pickle
import joblib
import numpy as np
import requests
from PIL import Image

image = Image.open('./img/spam.png')
st.set_page_config(page_title='Spam Detection', page_icon=image)

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# define a function to calculate the average word embedding for each row
def get_mean_word_embedding(row, w2v_model):
    words = row.split()
    embeddings = []
    for word in words:
        if word in w2v_model.wv:
            embeddings.append(w2v_model.wv[word])
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros((100,))
    

# load the saved objects
cv = pickle.load(open('count_vectorizer.pkl', 'rb'))

model = joblib.load(open('spam_detection_model_LR','rb'))

with open('word2vec.pkl', 'rb') as f:
    w2v_model = pickle.load(f)



st.title("Spam Detection System")
st.write("\t\t\tV 1.0")
st.write('---')

animation_spam = load_lottie('https://assets7.lottiefiles.com/private_files/lf30_prqvme9e.json')

with st.container():
    right_column, left_column = st.columns(2)
    with right_column:
        # Enter the message you want to classify
        new_message = st.text_area('Enter your message :', height=250)
    
    with left_column:
        st_lottie(animation_spam, speed=1, height=250, key="secoend")

if st.button('Predict'):

    # transform the new message using the loaded CountVectorizer object
    new_bow_features = cv.transform([new_message])
    
    # concatenate the bag-of-words features with the word embedding features
    new_w2v_features = get_mean_word_embedding(new_message, w2v_model)
    new_features = np.concatenate((new_bow_features.toarray(), np.expand_dims(new_w2v_features, axis=0)), axis=1)

    # Predict
    result = model.predict(new_features)
    # Display
    if result == 1:
        st.error("Spam")
    elif result == 0:
        st.success("Ham 'Not Spam'")


st.write('---')
st.write('')

animation_contact = load_lottie("https://assets5.lottiefiles.com/packages/lf20_ejryvjev.json")
with st.container():
    right_column, left_column = st.columns(2)
    with right_column:

        st.write('')
        st.write('')
        st.write('')
        st.write('_For any issue contact me via :_')

        '''

            [![Github](https://skillicons.dev/icons?i=github)](https://github.com/aliabdallah7)
            &nbsp;&nbsp;
            [![LinkedIn](https://skillicons.dev/icons?i=linkedin)](https://www.linkedin.com/in/ali-abdallah7/)
            &nbsp;&nbsp;
            [![Gmail](https://img.icons8.com/color/70/gmail-new.png)](mailto:ali.abdallah43792@gmail.com)

        '''

    with left_column:
        st_lottie(animation_contact, speed=1, height=200, key="third")


footer = """<style>
div.stButton > button:first-child {
  padding: 10px 20px;
  border: 2px solid #ff4d4d;
  background-color: #ffcdd2;
  color: #c62828;
  font-size: 16px;
  font-weight: bold;
  border-radius: 5px;
  box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.5);
  transition: all 0.2s ease;
}
div.stButton > button:hover:first-child {
  background-color: #ff4d4d;
  color: #fff;
  box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.8);
}
div.stButton > button:active:first-child {
  background-color: #c62828;
  color: #fff;
  box-shadow: inset 0px 2px 5px rgba(0, 0, 0, 0.5);
}

header {visibility: hidden;}

/* Light mode styles */
.my-paragraph {
  color: black;
}

/* Dark mode styles */
@media (prefers-color-scheme: dark) {
  .my-paragraph {
    color: white;
  }
}

a:link , a:visited{
color: #5C5CFF;
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

:root {
  --footer-bg-color: #333;
}

@media (prefers-color-scheme: dark) {
  :root {
    --footer-bg-color: rgb(14, 17, 23);
  }
}

@media (prefers-color-scheme: light) {
  :root {
    --footer-bg-color: white;
  }
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: var(--footer-bg-color);
color: black;
text-align: center;
}

</style>
<div class="footer">
<p class="my-paragraph">&copy; 2023 <a href="https://www.linkedin.com/in/ali-abdallah7/"> Ali Abdallah</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)
