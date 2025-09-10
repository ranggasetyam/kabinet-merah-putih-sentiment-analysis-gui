# Import library yang dibutuhkan
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

# Konfigurasi halaman Streamlit
# Menetapkan logo aplikasi
try:
    im = Image.open("Logo UNDIP.png")
except FileNotFoundError:
    im = "📊"

# Pengaturan halaman
st.set_page_config(
    page_title="Prediksi Analisis Sentimen Kabinet Merah Putih",
    page_icon=im,
    layout="wide"
)

# Fungsi-fungsi tambahan
# Fungsi untuk memuat model dan tokenizer
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

# Fungsi untuk mengonversi DataFrame ke file Excel
def convert_df_to_excel(df):
    """Mengubah DataFrame menjadi file Excel di memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

# Fungsi untuk mengonversi DataFrame ke file CSV
def convert_df_to_csv(df):
    """Mengubah DataFrame menjadi file CSV di memory."""
    return df.to_csv(index=False).encode('utf-8')

# Fungsi untuk menampilkan dialog konfirmasi
def ask_confirmation(state_key):
    """Mengubah state untuk menampilkan dialog konfirmasi."""
    st.session_state[state_key] = True

# Fungsi untuk membatalkan konfirmasi
def cancel_confirmation(state_key):
    """Membatalkan aksi dan mengembalikan state ke awal."""
    st.session_state[state_key] = False

# Inisialisasi session state untuk menyimpan data antar tab
if 'source_files' not in st.session_state:
    st.session_state.source_files = {}
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}
if 'predicted_files' not in st.session_state:
    st.session_state.predicted_files = {}
if 'confirm_reset_source' not in st.session_state:
    st.session_state.confirm_reset_source = False
if 'confirm_reset_processed' not in st.session_state:
    st.session_state.confirm_reset_processed = False
if 'confirm_reset_predicted' not in st.session_state:
    st.session_state.confirm_reset_predicted = False

# Memuat model dan tokenizer
MODEL_PATH = "ranggasetyam/indobertweet-kabinet-merah-putih"
tokenizer, model = load_model(MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if model:
    model.to(device)
    model.eval()

# Tampilan header aplikasi
st.logo(im, size="large")
st.title("🏛️ Prediksi Sentimen Kabinet Merah Putih pada Media Sosial X")
st.markdown("**Dikembangkan oleh Rangga Setya Mahendra**")

# Menampilkan expander untuk informasi aplikasi dan model
with st.expander("ℹ️ Tentang Aplikasi dan Model"):
    st.markdown("""
    Aplikasi ini dirancang untuk melakukan analisis sentimen terhadap komentar publik mengenai **Kabinet Merah Putih** di media sosial X ke dalam tiga kategori, yaitu **positif, netral, dan negatif**. Aplikasi ini berfungsi sebagai alat yang mudah digunakan bagi siapa saja yang ingin memahami persepsi publik terhadap kabinet baru, baik melalui data yang mereka kumpulkan sendiri maupun dengan melakukan *scraping* data secara langsung dari aplikasi ini.
    """)
    st.markdown("""
    Model yang digunakan untuk melakukan prediksi sentimen adalah **IndoBERTweet**, sebuah varian dari model BERT yang secara khusus dilakukan _pre-training_ pada jutaan data _tweet_ berbahasa Indonesia. Model ini telah melalui proses _fine-tuning_ pada _dataset_ komentar spesifik tentang Kabinet Merah Putih sehingga memiliki pemahaman konteks yang lebih baik.
    
    **Akurasi Model**: Model hasil _fine-tuning_ ini berhasil mencapai akurasi sebesar **83,76%** pada data pengujian.
    """)
    st.warning("""
    **Ketentuan Penggunaan Data**
    1. **Data *Upload***: Jika Anda menggunakan data sendiri, pastikan file berformat **Excel (.xlsx)** atau **CSV (.csv)** dan memiliki setidaknya satu kolom yang berisi teks komentar.
    2. **Data *Scraping***: Proses *scraping* memerlukan *auth_token* dari akun X Anda. Harap gunakan fitur ini dengan bijak dan sesuai dengan ketentuan layanan X.
    3. ***Pre-Processing***: Untuk hasil prediksi terbaik, sangat disarankan untuk melakukan semua langkah *pre-processing* yang tersedia.
    """)

# Membuat tab untuk setiap tahapan proses
tab1, tab2, tab3 = st.tabs(["**📁 Persiapan Data**", "**⚙️ *Pre-Processing***", "**📊 Prediksi Sentimen**"])

# Tab 1: Persiapan Data
with tab1:
    st.header("Persiapan Data Komentar")
    
    # Bagian untuk mengunggah file
    uploaded_files = st.file_uploader(
        "*Upload* file Excel (.xlsx) atau CSV (.csv) Anda.", 
        type=["xlsx", "csv"],
        accept_multiple_files=True
    )
    
    # Loop untuk memproses setiap file yang diunggah
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                if uploaded_file.name not in st.session_state.source_files:
                    # Membaca file berdasarkan ekstensi
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    # Menyimpan DataFrame ke session state
                    st.session_state.source_files[uploaded_file.name] = df
                    st.success(f"File '{uploaded_file.name}' berhasil ditambahkan.")
            except Exception as e:
                st.error(f"Gagal membaca file {uploaded_file.name}. Error: {e}")

    # Pemisah antara upload dan scraping
    st.markdown("<h4 style='text-align: center;'>atau</h4>", unsafe_allow_html=True)
    
    st.header("*Scraping* Data dari Media Sosial X")

    # Informasi tentang scraping
    with st.expander("ℹ️ Tentang *Scraping* Data"):
        st.markdown("""
        ***Scraping*** data dilakukan menggunakan **Tweet Harvest** yang dikembangkan oleh Helmi Satria. Tweet Harvest adalah sebuah *command-line tool* yang menggunakan Playwright untuk mengambil komentar dari hasil pencarian X berdasarkan kata kunci dan filter tertentu.
        """)
        st.info("""
        **Tweet Harvest** memerlukan auth_token dari X untuk mengakses dan menavigasi halaman pencarian X. auth_token dapat diperoleh dengan masuk ke X di *browser* Anda dan mengekstrak *cookie* auth_token.
        """)

    # Cek apakah aplikasi dijalankan di Streamlit Cloud
    url = streamlit_js_eval(js_expressions='window.location.href', want_output=True, key="GET_URL")
    is_cloud = False
    if url and ".streamlit.app" in url:
        is_cloud = True
        st.warning("Fitur *scraping* dinonaktifkan di Streamlit Cloud. Silakan jalankan aplikasi ini secara lokal untuk menggunakannya.")

    # Form untuk input parameter scraping
    with st.form("scraping_form"):
        twitter_auth_token = st.text_input("X Auth Token", type="password", help="Masukkan auth_token dari *cookie browser* Anda.", disabled=is_cloud)
        filename_input = st.text_input("Nama File *Output* (selalu .csv)", value="data_komentar.csv", disabled=is_cloud)
        search_keyword = st.text_input("Kata Kunci Pencarian", value="Kabinet Merah Putih", disabled=is_cloud)
        limit = st.number_input("Batas Jumlah Komentar", min_value=1, max_value=1000, value=100, step=10, disabled=is_cloud)
        submitted = st.form_submit_button("🔍 Mulai *Scraping*", disabled=is_cloud, type="primary")

    # Proses scraping setelah tombol ditekan
    if submitted:
        if not twitter_auth_token:
            st.error("Auth Token tidak boleh kosong!")
        else:
            filename = f"{filename_input.replace('.csv', '')}.csv"
            command = [
                "npx", "-y", "tweet-harvest@latest", "-o", filename,
                "-s", search_keyword, "--tab", "LATEST", "-l", str(limit),
                "--token", twitter_auth_token
            ]
            st.info("Proses *scraping* sedang berjalan...")
            log_placeholder = st.empty()
            log_output = ""
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', shell=True)
                # Menampilkan log proses scraping secara real-time
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
                    scraped_df = pd.read_csv(output_path)
                    st.session_state.source_files[filename] = scraped_df
                    st.success(f"Scraping selesai! File '{filename}' ditambahkan ke daftar data. Total data terkumpul: **{len(scraped_df)}**")
                else:
                    st.error("Scraping gagal. Tidak ada file yang dihasilkan.")
            except FileNotFoundError:
                st.session_state.source_files = None
                st.error("[WinError 2] Perintah 'npx' tidak ditemukan. Pastikan Node.js terinstall dan PATH sudah benar.")
            except Exception as e:
                st.error(f"Terjadi error saat scraping: {e}")

    # Menampilkan daftar file yang telah diupload atau di-scrap
    st.write("---")
    st.subheader("Daftar Data Komentar")

    if not st.session_state.source_files:
        st.info("Belum ada data yang di-*upload* atau di-*scrap*.")
    else:
        for filename, df in st.session_state.source_files.items():
            # Menampilkan data dalam expander
            with st.expander(f"`{filename}` ({len(df)} baris)"):
                # Menampilkan DataFrame di dalam expander
                st.dataframe(df, width='stretch', height=210)

                # Tombol unduh untuk setiap file
                col1, col2 = st.columns(2)
                col1.download_button(
                    "📥 Download (.csv)", 
                    convert_df_to_csv(df), 
                    f"{os.path.splitext(filename)[0]}.csv", 
                    "text/csv", 
                    key=f"source_csv_{filename}"
                )
                col2.download_button(
                    "📥 Download (.xlsx)", 
                    convert_df_to_excel(df), 
                    f"{os.path.splitext(filename)[0]}.xlsx", 
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                    key=f"source_xlsx_{filename}"
                )

        # Tombol untuk menghapus semua data sumber
        if not st.session_state.confirm_reset_source:
            st.button(
                "🔄 Hapus Semua Data Komentar", 
                on_click=ask_confirmation, 
                args=('confirm_reset_source',),
                help="Menghapus semua data komentar'.",
                key="reset_source_button",
                disabled=not st.session_state.source_files
            )
        else:
            st.warning("Apakah Anda yakin? **Semua data komentar akan hilang.**")
            
            if st.button("✅ Ya, Hapus", type="primary", key="confirm_source"):
                st.session_state.source_files = {}
                st.session_state.confirm_reset_source = False
                st.success("Semua data komentar telah dihapus.")
                st.rerun()

            st.button("❌ Batal", on_click=cancel_confirmation, args=('confirm_reset_source',), key="cancel_source")

# Tab 2: Pre-Processing
with tab2:
    st.header("Pembersihan dan Normalisasi Data")

    if not st.session_state.source_files:
        st.warning("Silakan *upload* atau *scraping* data terlebih dahulu di *tab* 'Persiapan Data'.")
    else:
        # Memilih file untuk diproses
        file_to_process_name = st.selectbox(
            "Pilih data yang akan diproses:",
            options=list(st.session_state.source_files.keys())
        )
        
        # Menampilkan data yang dipilih dan opsi pre-processing
        if file_to_process_name:
            df_to_process = st.session_state.source_files[file_to_process_name].copy()
            st.info(f"Data yang dipilih: **{file_to_process_name}** ({len(df_to_process)} baris)")
            st.dataframe(df_to_process, width='stretch', height=210)

            # Bagian untuk memilih kolom teks
            columns = df_to_process.columns.tolist()
            text_column_options = [col for col in columns if df_to_process[col].dtype == 'object']
            selected_column = st.selectbox("Pilih kolom yang berisi teks komentar:", text_column_options, index=None)

            if not selected_column:
                disable_process = True
                st.error("Kolom teks harus dipilih untuk melanjutkan *pre-processing*.")

            st.markdown("---")

            # Pilihan checkbox untuk langkah pre-processing
            st.subheader("Pilih Langkah *Pre-Processing*:")
            col1, col2 = st.columns(2)
            with col1:
                use_case_folding = st.checkbox("*Case Folding* (mengubah jadi huruf kecil)", value=True)
                use_text_cleaning = st.checkbox("*Text Cleaning* (hapus URL, mention, dll)", value=True)
            with col2:
                use_normalization = st.checkbox("*Normalization* (perbaiki kata slang)", value=True)
                use_duplicate_handling = st.checkbox("*Duplicate Handling* (hapus data duplikat)", value=True)

            # Tombol untuk memulai pre-processing
            if st.button("✨ Mulai *Pre-Processing*", type="primary", disabled=not selected_column):
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
                
                    df_asli_tanpa_duplikat = df_to_process.loc[processed_df.index]
                    df_gabungan = df_asli_tanpa_duplikat.copy()
                    df_gabungan['processed_text'] = comparison_df['processed_text']

                    base_name = os.path.splitext(file_to_process_name)[0]
                    processed_file_name = f"processed_{base_name}"

                    st.session_state.processed_files[processed_file_name] = {
                        "comparison": comparison_df,
                        "full": df_gabungan
                    }
                    
                    st.success(f"*Pre-processing* selesai! Data baru hasil *pre-processing* telah ditambahkan ke daftar di bawah.")

        # Menampilkan daftar hasil pre-processing
        st.write("---")
        st.subheader("Daftar Data Hasil *Pre-Processing*")

        if not st.session_state.processed_files:
            st.info("Belum ada data yang diproses.")
        else:
            for base_name, data_group in st.session_state.processed_files.items():
                with st.expander(f"`{base_name}`"):
                    
                    # Menampilkan data perbandingan
                    if 'comparison' in data_group:
                        df_comp = data_group['comparison']
                        st.markdown(f"#### Data Perbandingan ({len(df_comp)} baris)")
                        st.caption("Hanya berisi kolom teks asli dan teks yang telah dilakukan *pre-processing*.")
                        st.dataframe(df_comp, width='stretch', height=210)
                        col1, col2 = st.columns(2)
                        col1.download_button("📥 Download (.csv)", convert_df_to_csv(df_comp), f"processed_comparison_{base_name}.csv", "text/csv", key=f"csv_proc_comp_{base_name}")
                        col2.download_button("📥 Download (.xlsx)", convert_df_to_excel(df_comp), f"processed_comparison_{base_name}.xlsx", key=f"xlsx_proc_comp_{base_name}")

                    # Menampilkan data gabungan
                    if 'full' in data_group:
                        df_full = data_group['full']
                        st.markdown(f"#### Data Gabungan ({len(df_full)} baris)")
                        st.caption("Berisi semua kolom asli ditambah kolom teks yang telah dilakukan *pre-processing*.")
                        st.dataframe(df_full, width='stretch', height=210)
                        col3, col4 = st.columns(2)
                        col3.download_button("📥 Download (.csv)", convert_df_to_csv(df_full), f"processed_{base_name}.csv", "text/csv", key=f"csv_proc_full_{base_name}")
                        col4.download_button("📥 Download (.xlsx)", convert_df_to_excel(df_full), f"processed_{base_name}.xlsx", key=f"xlsx_proc_full_{base_name}")
 
            # Tombol untuk menghapus semua data hasil pre-processing
            if not st.session_state.confirm_reset_processed:
                st.button(
                    "🔄 Hapus Semua Hasil *Pre-Processing*", 
                    on_click=ask_confirmation, 
                    args=('confirm_reset_processed',),
                    help="Menghapus semua data hasil *Pre-Processing*'.",
                    key="reset_processed_button",
                    disabled=not st.session_state.processed_files
                )
            else:
                st.warning("Apakah Anda yakin? **Semua data hasil *pre-processing* akan hilang.**")
                
                if st.button("✅ Ya, Hapus", type="primary", key="confirm_processed"):
                    st.session_state.processed_files = {}
                    st.session_state.confirm_reset_processed = False
                    st.success("Semua data hasil *pre-processing* telah dihapus.")
                    st.rerun()

                st.button("❌ Batal", on_click=cancel_confirmation, args=('confirm_reset_processed',), key="cancel_processed")

# Tab 3: Prediksi Sentimen
with tab3:
    st.header("Prediksi Sentimen")
    
    if not st.session_state.processed_files and not st.session_state.predicted_files:
        st.warning("Silakan lakukan *pre-processing* data terlebih dahulu di *tab* '*Pre-Processing*'.")
    else:
        # Memuat data hasil pre-processing untuk diprediksi
        processed_full_files = {}
        for name, data_group in st.session_state.processed_files.items():
            if 'full' in data_group:
                processed_full_files[name] = data_group['full']
        
        if not processed_full_files:
             st.warning("Tidak ada data hasil *pre-processing* yang tersedia untuk diprediksi.")
        else:
            # Memilih file untuk diprediksi
            file_to_predict_name = st.selectbox(
                "Pilih data yang akan diprediksi:",
                options=list(processed_full_files.keys())
            )

            # Menampilkan data yang dipilih
            if file_to_predict_name:
                df_to_predict = processed_full_files[file_to_predict_name].copy()
                st.info(f"Data yang dipilih: **{file_to_predict_name}** ({len(df_to_predict)} baris)")
                st.dataframe(df_to_predict, width='stretch', height=210)
                show_probs = st.checkbox("Tampilkan probabilitas untuk setiap kelas sentimen")
            
                # Tombol untuk memulai prediksi
                if st.button("🚀 Mulai Prediksi Sentimen", type="primary"):
                    if not model or not tokenizer:
                        st.error("Model tidak berhasil dimuat. Prediksi tidak dapat dilanjutkan.")
                    else:
                        with st.spinner("Prediksi sentimen sedang berjalan... Harap tunggu."):
                            texts_to_predict = df_to_predict['processed_text'].tolist()
                            
                            # Mengubah teks menjadi format yang sesuai untuk model
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

                            # Melakukan prediksi dengan model
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

                            # Mengonversi label numerik ke label teks
                            labels_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'} 
                            predicted_labels = [labels_map[p] for p in all_predictions]

                            # Menambahkan hasil prediksi ke DataFrame
                            result_comparison = df_to_predict[['processed_text']].copy()
                            result_comparison['predicted_sentiment'] = predicted_labels
                            
                            result_full = df_to_predict.copy()
                            result_full['predicted_sentiment'] = predicted_labels
                            
                            # Menambahkan probabilitas jika checkbox dicentang
                            if show_probs:
                                all_probs_flat = np.concatenate(all_probs, axis=0)
                                result_comparison['prob_negatif'] = all_probs_flat[:, 0]
                                result_comparison['prob_netral'] = all_probs_flat[:, 1]
                                result_comparison['prob_positif'] = all_probs_flat[:, 2]

                                result_full['prob_negatif'] = all_probs_flat[:, 0]
                                result_full['prob_netral'] = all_probs_flat[:, 1]
                                result_full['prob_positif'] = all_probs_flat[:, 2]

                                original_base_name = file_to_predict_name.replace("processed_", "")
                                prediction_name = f"predicted_{original_base_name}"
                                                        
                                st.session_state.predicted_files[prediction_name] = {
                                    "comparison": result_comparison,
                                    "full": result_full
                                }
                                
                                st.success(f"Prediksi selesai! Data baru hasil prediksi sentimen telah ditambahkan ke daftar di bawah.")

        # Menampilkan daftar hasil prediksi
        st.write("---")
        st.subheader("Daftar Data Hasil Prediksi")

        if not st.session_state.predicted_files:
            st.info("Belum ada data yang diprediksi.")
        else:
            for prediction_name, data_group in st.session_state.predicted_files.items():
                with st.expander(f"`{prediction_name}`"):
                    
                    # Menampilkan data perbandingan hasil prediksi
                    if 'comparison' in data_group:
                        df_comp = data_group['comparison']
                        st.markdown(f"#### Hasil Prediksi ({len(df_comp)} baris)")
                        st.caption("Hanya berisi kolom teks yang telah dilakukan *pre-processing* dan hasil prediksi sentimen.")
                        st.dataframe(df_comp, width='stretch', height=210)
                        col1, col2 = st.columns(2)
                        col1.download_button("📥 Download (.csv)", convert_df_to_csv(df_comp), f"predicted_comparison_{prediction_name}.csv", "text/csv", key=f"csv_pred_comp_{prediction_name}")
                        col2.download_button("📥 Download (.xlsx)", convert_df_to_excel(df_comp), f"predicted_comparison_{prediction_name}.xlsx", key=f"xlsx_pred_comp_{prediction_name}")

                    # Menampilkan data gabungan hasil prediksi
                    if 'full' in data_group:
                        df_full = data_group['full']
                        st.markdown(f"#### Data Gabungan ({len(df_full)} baris)")
                        st.caption("Berisi semua kolom asli ditambah kolom teks yang telah dilakukan *pre-processing* dan hasil prediksi sentimen.")
                        st.dataframe(df_full, width='stretch', height=210)
                        col3, col4 = st.columns(2)
                        col3.download_button("📥 Download (.csv)", convert_df_to_csv(df_full), f"full_predicted_{prediction_name}.csv", "text/csv", key=f"csv_pred_full_{prediction_name}")
                        col4.download_button("📥 Download (.xlsx)", convert_df_to_excel(df_full), f"full_predicted_{prediction_name}.xlsx", key=f"xlsx_pred_full_{prediction_name}")
                        
                        # Visualisasi hasil prediksi
                        st.markdown("---")
                        st.markdown("#### Visualisasi")
                        sentiment_counts = df_full['predicted_sentiment'].value_counts()
                        urutan_label = ['Positif', 'Netral', 'Negatif']
                        sentiment_counts = sentiment_counts.reindex(urutan_label, fill_value=0)
                        
                        fig, ax = plt.subplots(figsize=(3.5, 3))
                        colors = {'Positif': '#34A853', 'Netral': '#4285F4', 'Negatif': '#EA4335'}
                        bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=[colors.get(x, '#808080') for x in sentiment_counts.index])
                        
                        ax.set_title('Distribusi Hasil Prediksi Sentimen', fontsize=8, pad=8)
                        ax.set_ylabel('Jumlah Komentar', fontsize=7)
                        ax.set_xlabel('Sentimen', fontsize=7)
                        ax.tick_params(axis='both', labelsize=7)
                        ax.grid(axis='y', linestyle='--', alpha=0.25)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{int(height)}', ha='center', va='bottom', fontsize=8)
                        
                        st.pyplot(fig, width='content')

                        img_buffer = io.BytesIO()
                        fig.savefig(img_buffer, format="png", bbox_inches="tight")
                        img_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download Visualisasi (.png)",
                            data=img_buffer,
                            file_name=f"visualisasi_sentimen_{prediction_name}.png",
                            mime="image/png"
                        )

            # Tombol untuk menghapus semua data hasil prediksi
            if not st.session_state.confirm_reset_predicted:
                st.button(
                    "🔄 Hapus Semua Hasil Prediksi", 
                    on_click=ask_confirmation, 
                    args=('confirm_reset_predicted',),
                    help="Menghapus semua data hasil prediksi'.",
                    key="reset_predicted_button",
                    disabled=not st.session_state.predicted_files
                )
            else:
                st.warning("Apakah Anda yakin? **Semua data hasil prediksi sentimen akan hilang.**")
                
                if st.button("✅ Ya, Hapus", type="primary", key="confirm_predicted"):
                    st.session_state.predicted_files = {}
                    st.session_state.confirm_reset_predicted = False
                    st.success("Semua data hasil prediksi telah dihapus.")
                    st.rerun()

                st.button("❌ Batal", on_click=cancel_confirmation, args=('confirm_reset_predicted',), key="cancel_predicted")