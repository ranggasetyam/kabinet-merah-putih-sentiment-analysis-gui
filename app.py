import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import subprocess
import os
import io
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from streamlit_js_eval import streamlit_js_eval

# =====================================================================================
# KONFIGURASI HALAMAN STREAMLIT
# =====================================================================================
im = Image.open("Logo UNDIP.png")

st.set_page_config(
    page_title="Prediksi Analisis Sentimen Kabinet Merah Putih",
    page_icon=im,
    layout="wide"
)

# =====================================================================================
# FUNGSI-FUNGSI BANTUAN
# =====================================================================================

# Cache untuk memuat model agar tidak perlu di-load ulang setiap kali ada interaksi
@st.cache_resource
def load_model(model_path):
    """Memuat model dan tokenizer."""
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error memuat model: {e}. Pastikan folder 'model_terbaik' ada di direktori yang sama dengan app.py.")
        return None, None

def convert_df_to_excel(df):
    """Mengubah DataFrame menjadi file Excel di memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

# =====================================================================================
# INISIALISASI MODEL DAN SESSION STATE
# =====================================================================================
# Inisialisasi session state untuk menyimpan data antar tab
if 'data_asli' not in st.session_state:
    st.session_state.data_asli = None
if 'data_source_name' not in st.session_state:
    st.session_state.data_source_name = ""
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = None
if 'data_predicted' not in st.session_state:
    st.session_state.data_predicted = None
if 'scraped_df' not in st.session_state:
    st.session_state.scraped_df = None

# Memuat model di awal
MODEL_PATH = "ranggasetyam/indobertweet-kabinet-merah-putih"
tokenizer, model = load_model(MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if model:
    model.to(device)
    model.eval()

# =====================================================================================
# TAMPILAN UTAMA GUI
# =====================================================================================
st.logo(im, size="large")
st.title("🏛️ Prediksi Analisis Sentimen Kabinet Merah Putih pada Media Sosial X")
st.markdown("**Dikembangkan oleh Rangga Setya Mahendra**")

with st.expander("ℹ️ Tentang Aplikasi dan Model"):
    st.markdown("""
    Aplikasi ini dirancang untuk melakukan analisis sentimen terhadap komentar publik mengenai **Kabinet Merah Putih** di media sosial X. Dengan memanfaatkan kekuatan model pemrosesan bahasa alami (NLP) canggih, aplikasi ini dapat mengklasifikasikan opini ke dalam tiga kategori: **Positif, Netral, dan Negatif**.
    
    Tujuan dari GUI ini adalah untuk menyediakan alat yang mudah digunakan bagi siapa saja yang ingin memahami persepsi publik terhadap kabinet baru, baik melalui data yang mereka kumpulkan sendiri maupun dengan melakukan *scraping* data secara langsung dari aplikasi ini.
    """)
    st.subheader("Model yang Digunakan")
    st.markdown("""
    Model yang menjadi inti dari aplikasi ini adalah **IndoBERTweet**, sebuah varian dari model BERT yang secara khusus di-_pre-trained_ pada jutaan data _tweet_ berbahasa Indonesia. Model ini telah melalui proses _fine-tuning_ pada dataset komentar spesifik tentang Kabinet Merah Putih, sehingga memiliki pemahaman konteks yang lebih baik.
    
    - **Akurasi Model**: Model hasil _fine-tuning_ ini berhasil mencapai akurasi sebesar **83.76%** pada data pengujian.
    - **Kemampuan**: Model ini sangat andal dalam membedakan sentimen **Positif** dan **Negatif**. Namun, seperti model NLP pada umumnya, ia mungkin menghadapi tantangan pada sentimen **Netral** yang seringkali ambigu atau mengandung kalimat sarkasme.
    """)
    st.warning("""
    **Ketentuan Penggunaan Data**
    1. **Data Upload**: Jika Anda menggunakan data sendiri, pastikan file berformat **Excel (.xlsx)** dan memiliki setidaknya satu kolom yang berisi teks komentar.
    2. **Data Scraping**: Proses *scraping* memerlukan *auth_token* dari akun X Anda. Harap gunakan fitur ini dengan bijak dan sesuai dengan ketentuan layanan X.
    3. **Pre-Processing**: Untuk hasil prediksi terbaik, sangat disarankan untuk melakukan semua langkah *pre-processing* yang tersedia.
    """)

tab1, tab2, tab3 = st.tabs(["**📥 Persiapan Data**", "**⚙️ Pre-Processing**", "**📊 Prediksi Sentimen**"])

# =====================================================================================
# TAB 1: DATA PREPARATION
# =====================================================================================
with tab1:
    st.header("Persiapan Data Komentar")
    
    uploaded_file = st.file_uploader("*Upload* file Excel Anda yang berisi daftar komentar.", type=["xlsx"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.data_asli = df
            st.session_state.scraped_df = None # Reset scraped data
            st.session_state.data_source_name = uploaded_file.name
            st.success(f"File '{uploaded_file.name}' berhasil di-upload. Total {len(df)} baris data.")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Gagal membaca file Excel. Pastikan format file sudah benar. Error: {e}")

    st.markdown("<h4 style='text-align: center;'>atau</h4>", unsafe_allow_html=True)
    
    st.subheader("*Scraping* Data dari Media Sosial X")

    url = streamlit_js_eval(js_expressions='window.location.href', want_output=True, key="GET_URL")
    is_cloud = False
    if url and ".streamlit.app" in url:
        is_cloud = True
        st.warning("Anda menjalankan aplikasi ini di Streamlit Cloud. Fitur *scraping* X tidak tersedia di lingkungan ini karena keterbatasan dari Streamlit Cloud. Silakan jalankan aplikasi ini secara lokal di komputer Anda (pastikan telah terpasang Node.js (LTS)) untuk menggunakan fitur *scraping*.")

    with st.expander("ℹ️ Panduan *Scraping* Data"):
        st.markdown("""
        ***Scraping*** data dilakukan menggunakan **Tweet Harvest** yang dikembangkan oleh Helmi Satria. Tweet Harvest adalah sebuah *command-line tool* yang menggunakan Playwright untuk mengambil komentar dari hasil pencarian X berdasarkan kata kunci dan filter tertentu.
        """)
        st.info("""
        **Tweet Harvest** memerlukan auth_token dari X untuk mengakses dan menavigasi halaman pencarian X. auth_token dapat diperoleh dengan masuk ke X di *browser* Anda dan mengekstrak *cookie* auth_token.
        """)
    
    with st.form("scraping_form"):
        twitter_auth_token = st.text_input("X Auth Token", type="password", help="Masukkan auth_token dari *cookie browser* Anda.", disabled=is_cloud)
        filename = st.text_input("Nama File *Output* (selalu .csv)", value="data_komentar.csv", disabled=is_cloud)
        search_keyword = st.text_input("Kata Kunci Pencarian", value="Kabinet Merah Putih", disabled=is_cloud)
        limit = st.number_input("Batas Jumlah Komentar", min_value=10, max_value=2000, value=100, step=10, disabled=is_cloud)

        if is_cloud:
            st.warning(
                "Fitur *scraping* dinonaktifkan saat aplikasi berjalan di Streamlit Cloud. Silakan jalankan aplikasi secara lokal untuk menggunakan fitur ini."
            )

        submitted = st.form_submit_button("🔍 Mulai Scraping", disabled=is_cloud)

    if submitted:
        if not twitter_auth_token:
            st.error("Auth Token tidak boleh kosong!")
        else:
            st.session_state.scraped_df = None 
            command = [
                "npx", "-y", "tweet-harvest@latest",
                "-o", filename,
                "-s", search_keyword,
                "--tab", "LATEST",
                "-l", str(limit),
                "--token", twitter_auth_token
            ]
            st.info("Proses *scraping* sedang berjalan... Harap tunggu.")
            log_placeholder = st.empty()
            log_output = ""
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', shell=True)
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        log_output += output.strip() + "\n"
                        log_placeholder.code(log_output, language='bash')
                process.wait()
                output_path = os.path.join("tweets-data", filename)
                if os.path.exists(output_path):
                    scraped_df_result = pd.read_csv(output_path)
                    st.session_state.data_asli = scraped_df_result
                    st.session_state.scraped_df = scraped_df_result
                    st.session_state.data_source_name = filename
                else:
                    st.session_state.scraped_df = None
                    st.error("*Scraping* gagal. Tidak ada file yang dihasilkan. Cek log di atas untuk detail error.")
            except FileNotFoundError:
                st.session_state.scraped_df = None
                st.error("[WinError 2] Perintah 'npx' tidak ditemukan. Pastikan Node.js terinstall dan PATH sudah benar.")
            except Exception as e:
                st.session_state.scraped_df = None
                st.error(f"Terjadi error saat menjalankan scraping: {e}")

    if st.session_state.scraped_df is not None:
        st.success(f"*Scraping* data telah selesai! Total data terkumpul: **{len(st.session_state.scraped_df)}**")
        st.dataframe(st.session_state.scraped_df)
        excel_data = convert_df_to_excel(st.session_state.scraped_df)
        st.download_button(
            label="📥 *Download* Data",
            data=excel_data,
            file_name=st.session_state.data_source_name.replace('.csv', '.xlsx'),
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

# =====================================================================================
# TAB 2: PRE-PROCESSING
# =====================================================================================
with tab2:
    st.header("Pembersihan dan Normalisasi Data")

    if st.session_state.data_asli is None:
        st.warning("Silakan *upload* atau ambil data terlebih dahulu di *tab* 'Data Preparation'.")
    else:
        st.info(f"Data yang digunakan: **{st.session_state.data_source_name}** ({len(st.session_state.data_asli)} baris)")
        
        df_to_process = st.session_state.data_asli.copy()
        st.dataframe(df_to_process)

        # Opsi memilih kolom
        columns = df_to_process.columns.tolist()
        text_column_options = [col for col in columns if df_to_process[col].dtype == 'object']
        selected_column = st.selectbox("Pilih kolom yang berisi teks komentar:", text_column_options)

        st.markdown("---")
        st.subheader("Pilih Langkah *Pre-Processing*:")
        col1, col2 = st.columns(2)
        with col1:
            use_case_folding = st.checkbox("Case Folding (mengubah jadi huruf kecil)", value=True)
            use_text_cleaning = st.checkbox("Text Cleaning (hapus URL, mention, dll.)", value=True)
        with col2:
            use_normalization = st.checkbox("Normalization (perbaiki kata slang)", value=True)
            use_duplicate_handling = st.checkbox("Duplicate Handling (hapus data duplikat)", value=True)

        if st.button("✨ Mulai *Pre-Processing*"):
            with st.spinner("*Pre-processing* sedang berjalan... Harap tunggu."):
                processed_df = df_to_process.copy()
                processed_df['original_text'] = processed_df[selected_column]
                
                # 1. Case Folding
                if use_case_folding:
                    processed_df[selected_column] = processed_df[selected_column].astype(str).str.lower()

                # 2. Text Cleaning
                if use_text_cleaning:
                    def text_cleaning(text):
                        text = re.sub(r'#\w+', '', text)
                        text = re.sub(r'@[A-Za-z0-9_]+', '', text)
                        text = re.sub(r'RT[\s]+', '', text)
                        text = re.sub(r'https?://\S+', '', text)
                        text = re.sub(r'[^A-Za-z0-9\s!?]', '', text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        return text
                    processed_df[selected_column] = processed_df[selected_column].apply(text_cleaning)

                # 3. Normalization
                if use_normalization:
                    try:
                        kamus_df = pd.read_csv('kamus_normalisasi.csv')
                        norm_dict = dict(zip(kamus_df.iloc[:, 0], kamus_df.iloc[:, 1]))
                        
                        def normalize_text(text):
                            return ' '.join([norm_dict.get(word, word) for word in text.split()])
                        
                        processed_df[selected_column] = processed_df[selected_column].apply(normalize_text)
                    except FileNotFoundError:
                        st.error("File 'kamus_normalisasi.csv' tidak ditemukan. Normalisasi dilewati.")
                    except Exception as e:
                        st.error(f"Error saat normalisasi: {e}")

                # 4. Duplicate Handling
                if use_duplicate_handling:
                    before_count = len(processed_df)
                    processed_df.drop_duplicates(subset=[selected_column], inplace=True)
                    after_count = len(processed_df)
                    st.info(f"{before_count - after_count} data duplikat telah dihapus.")

                comparison_df = processed_df[['original_text', selected_column]].rename(columns={selected_column: 'processed_text'})
                st.session_state.data_processed = comparison_df
            
            st.success("*Pre-processing* telah selesai!")
        
        if st.session_state.data_processed is not None:
            st.subheader("Perbandingan Teks Sebelum dan Sesudah *Pre-processing*")
            st.dataframe(st.session_state.data_processed)

            col_dl, col_reset = st.columns(2)
            with col_dl:
                excel_processed = convert_df_to_excel(st.session_state.data_processed)
                st.download_button(
                    label="📥 Download Data Hasil *Pre-processing*",
                    data=excel_processed,
                    file_name="processed_data.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
            with col_reset:
                if st.button("🔄 Reset Proses"):
                    st.session_state.data_processed = None
                    st.rerun()

# =====================================================================================
# TAB 3: PREDICTION
# =====================================================================================
with tab3:
    st.header("Prediksi Sentimen")
    
    if st.session_state.data_processed is None:
        st.warning("Silakan lakukan *pre-processing* data terlebih dahulu di *tab* 'Pre-Processing'.")
    else:
        st.info(f"Data yang akan diprediksi: **{len(st.session_state.data_processed)}** baris data yang sudah dilakukan *pre-processing*.")
        
        show_probs = st.checkbox("Tampilkan probabilitas untuk setiap kelas sentimen")
        
        if st.button("🚀 Mulai Prediksi Sentimen"):
            if not model or not tokenizer:
                st.error("Model tidak berhasil dimuat. Prediksi tidak dapat dilanjutkan.")
            else:
                with st.spinner("Prediksi sentimen sedang berjalan... Harap tunggu."):
                    texts_to_predict = st.session_state.data_processed['processed_text'].tolist()
                    
                    max_len = 45
                    encoded_data = tokenizer.batch_encode_plus(
                        texts_to_predict,
                        add_special_tokens=True,
                        return_attention_mask=True,
                        padding='max_length',
                        max_length=max_len,
                        truncation=True,
                        return_tensors='pt'
                    )
                    
                    input_ids = encoded_data['input_ids']
                    attention_masks = encoded_data['attention_mask']

                    dataset = TensorDataset(input_ids, attention_masks)
                    dataloader = DataLoader(dataset, batch_size=32)

                    all_predictions = []
                    all_probs = []

                    with torch.no_grad():
                        for batch in dataloader:
                            b_input_ids, b_attention_mask = batch
                            b_input_ids = b_input_ids.to(device)
                            b_attention_mask = b_attention_mask.to(device)

                            outputs = model(b_input_ids, attention_mask=b_attention_mask)
                            logits = outputs.logits
                            
                            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                            all_probs.append(probs)

                            preds = np.argmax(probs, axis=1)
                            all_predictions.extend(preds)

                    labels_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'} 
                    predicted_labels = [labels_map[p] for p in all_predictions]

                    result_df = st.session_state.data_processed.copy()
                    result_df['predicted_sentiment'] = predicted_labels
                    
                    if show_probs:
                        all_probs_flat = np.concatenate(all_probs, axis=0)
                        result_df['prob_negatif'] = all_probs_flat[:, 0]
                        result_df['prob_netral'] = all_probs_flat[:, 1]
                        result_df['prob_positif'] = all_probs_flat[:, 2]

                    st.session_state.data_predicted = result_df
                
                sentiment_counts = result_df['predicted_sentiment'].value_counts()
                pos_count = sentiment_counts.get('Positif', 0)
                neu_count = sentiment_counts.get('Netral', 0)
                neg_count = sentiment_counts.get('Negatif', 0)
                st.success(f"Prediksi sentimen telah selesai! Total sentimen dari data sebanyak: **{pos_count}** komentar positif, **{neu_count}** komentar netral, dan **{neg_count}** komentar negatif.")

        if st.session_state.data_predicted is not None:
            st.subheader("Hasil Prediksi Sentimen")
            st.dataframe(st.session_state.data_predicted)
            
            excel_predicted = convert_df_to_excel(st.session_state.data_predicted)
            st.download_button(
                label="📥 Download Hasil Prediksi",
                data=excel_predicted,
                file_name="prediction_results.xlsx",
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

            st.markdown("---")
            st.header("Visualisasi Prediksi Sentimen")
            
            sentiment_counts = st.session_state.data_predicted['predicted_sentiment'].value_counts()
            
            urutan_label = ['Positif', 'Netral', 'Negatif']
            sentiment_counts = sentiment_counts.reindex(urutan_label, fill_value=0)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = {'Positif': '#34A853', 'Netral': '#4285F4', 'Negatif': '#EA4335'}
            bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=[colors.get(x, '#808080') for x in sentiment_counts.index])
            
            ax.set_title('Distribusi Hasil Prediksi Sentimen', fontsize=16, pad=20)
            ax.set_ylabel('Jumlah Komentar', fontsize=12)
            ax.set_xlabel('Sentimen', fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 3, f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            st.pyplot(fig)
