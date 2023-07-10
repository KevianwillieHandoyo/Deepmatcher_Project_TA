import streamlit as st
import deepmatcher as dm
import torch
import pandas as pd


model = dm.MatchingModel(attr_summarizer='rnn')
model.load_state('rnn_model2.pth', map_location=torch.device('cpu'))

import nltk
nltk.download('punkt')

def predictor(file):
    unlabeled = dm.data.process_unlabeled(
        path=file,
        trained_model=model,
        ignore_columns=('ltable_Unnamed: 0', 'rtable_Unnamed: 0'))

    predictions = model.run_prediction(unlabeled, output_attributes=True)
    predictions['match_prediction'] = predictions['match_score'].apply(lambda score: 1 if score >= 0.5 else 0)
    predictions = predictions[['match_score', 'match_prediction'] + predictions.columns.values[1:-1].tolist()]

    return predictions

def dropdown_list(data):
    drop_list = data['rtable_Nama_Produk'].unique()
    return drop_list

def main():
    st.set_page_config(page_title='Data Integration with Deepmatcher', layout='wide')
    with st.container():

       column1, column2, column3 = st.columns([1,7,1])
       with column1:
           st.image('https://si.its.ac.id/labs/spk/SimTA/assets/img/logo_lab/ADDI.png', width=120)
       with column2:
           st.markdown("<h1 style='text-align: center'>Tugas Akhir Sistem Informasi ITS</h1>", unsafe_allow_html=True)
       with column3:
           st.image('https://zedemy.com/wp-content/uploads/2020/09/Logo-SI.png', width=100)
    with st.container():
        #st.markdown("<h1 style='text-align: center'>Tugas Akhir Sistem Informasi ITS</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center'>INTEGRASI DATA PRODUK KOSMETIK BPOM DAN E-COMMERCE MENGGUNAKAN BIDIRECTIONAL NEURAL NETWORK</h2>", unsafe_allow_html=True)
        st.divider()
    with st.container():
        st.subheader('Contoh Data Hasil Prediksi Model BRNN')
        data_contoh = pd.read_csv('DEEPMATCHER_unlabeled_predictions.csv')
        st.dataframe(data_contoh, width=2000)
        st.divider()
    with st.container():
        st.subheader('Cari Produk E-Commerce')
        text_input = st.text_input(
            "Cari produk serbuk tabur e-commerce dengan memasukkan nama produk"
        )
        if text_input:
            rslt_df = data_contoh[(data_contoh['match_prediction'] == 1) & data_contoh['rtable_Nama_Produk'].str.contains(text_input, case=False)]
            st.dataframe(rslt_df, width=2000)
        st.divider()
    with st.container():
        st.subheader('Coba Prediksi Data')
        st.markdown('Silahkan upload data CSV di sini.')
        st.markdown('- Data CSV harus memiliki header **_id, ltable_Unnamed: 0, rtable_Unnamed: 0, ltable_Nama Produk, rtable_Merk, rtable_Nama Produk, ltable_Merk.**')
        uploadedfile = st.file_uploader("Upload CSV File", type="csv")
        csv_path = 'prediction.csv'
        if st.button("Prediksi Data"):
            if uploadedfile is not None:
                df = pd.read_csv(uploadedfile)
                df.to_csv(csv_path, encoding='utf-8')
                result = predictor(csv_path)
                st.dataframe(result, width=2000)
            #with NamedTemporaryFile(dir='.', suffix='.csv') as f:
                #f.write(uploadedfile.getbuffer())
                #result = predictor(f.name)
            #st.write(result)
            else:
                st.error('File belum diupload!')


if __name__ == '__main__':
    main()
