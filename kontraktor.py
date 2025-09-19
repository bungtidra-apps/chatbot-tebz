import streamlit as st
import openai
import PyPDF2
import io
import tiktoken # Untuk menghitung token, penting untuk manajemen API

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Penganalisis Kontrak PDF dengan OpenAI",
    page_icon="📄",
    layout="wide" # Menggunakan layout wide agar lebih banyak ruang
)

st.title("📄 Penganalisis Kontrak Kontrak PDF dengan OpenAI")
st.markdown("""
Aplikasi ini memungkinkan Anda mengunggah dokumen kontrak dalam format PDF dan mendapatkan analisis mendalam
menggunakan kekuatan model bahasa AI dari OpenAI. Masukkan API Key Anda di sidebar,
unggah kontrak, dan pilih jenis analisis yang Anda inginkan.
""")

# --- Sidebar untuk API Key dan Model Selection ---
with st.sidebar:
    st.header("🔑 Konfigurasi OpenAI")
    openai_api_key = st.text_input(
        "Masukkan OpenAI API Key Anda",
        type="password", # Menyembunyikan input untuk keamanan
        help="Anda bisa mendapatkan API key Anda dari: https://platform.openai.com/account/api-keys"
    )
    st.info("API Key Anda hanya digunakan untuk sesi ini dan tidak akan disimpan.")

    st.subheader("Pengaturan Model AI")
    model_choice = st.selectbox(
        "Pilih Model OpenAI:",
        ("gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"), # gpt-4o adalah yang terbaru dan seringkali terbaik
        help="Pilih model AI yang akan digunakan untuk analisis. gpt-4o umumnya memberikan hasil terbaik, gpt-3.5-turbo lebih murah dan cepat."
    )
    temperature = st.slider(
        "Kreativitas Model (Temperature)",
        min_value=0.0, max_value=1.0, value=0.7, step=0.1,
        help="Nilai yang lebih tinggi (mendekati 1.0) akan membuat output lebih kreatif dan bervariasi. Nilai yang lebih rendah (mendekati 0.0) akan membuat output lebih fokus dan deterministik."
    )
    max_tokens = st.slider(
        "Maksimum Token Output",
        min_value=500, max_value=4000, value=1500, step=100,
        help="Jumlah maksimum token yang diizinkan untuk respons AI. Sesuaikan berdasarkan panjang respons yang Anda harapkan."
    )

# --- Fungsi untuk Ekstraksi Teks dari PDF ---
@st.cache_data # Cache hasil ekstraksi agar tidak diulang jika file tidak berubah
def extract_text_from_pdf(pdf_file_bytes):
    """Mengekstrak teks dari objek file PDF."""
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_file_bytes))
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text() or "" # Gunakan string kosong jika tidak ada teks
    return text

# --- Fungsi untuk Menghitung Token ---
@st.cache_data # Cache hasil perhitungan token
def count_tokens(text, model_name):
    """Menghitung jumlah token dalam teks menggunakan tiktoken."""
    try:
        # Menyesuaikan nama model untuk tiktoken jika perlu (misal: gpt-4-turbo -> gpt-4)
        if "gpt-4o" in model_name:
            encoding_model = "gpt-4o"
        elif "gpt-4" in model_name:
            encoding_model = "gpt-4"
        elif "gpt-3.5" in model_name:
            encoding_model = "gpt-3.5-turbo"
        else:
            encoding_model = "cl100k_base" # Fallback umum

        encoding = tiktoken.encoding_for_model(encoding_model)
        return len(encoding.encode(text))
    except KeyError:
        # Fallback jika model tidak dikenal oleh tiktoken secara spesifik
        st.warning(f"Tiktoken tidak memiliki encoding spesifik untuk model '{model_name}'. Menggunakan perkiraan.")
        return len(text.split()) # Perkiraan kasar berdasarkan jumlah kata

# --- Fungsi untuk Analisis Kontrak dengan OpenAI ---
def analyze_contract_with_openai(contract_text, openai_key, model_name, prompt_template, temp, tokens):
    """Mengirim teks kontrak dan prompt ke OpenAI API untuk analisis."""
    if not openai_key:
        st.error("❌ Error: Silakan masukkan OpenAI API Key Anda di sidebar.")
        return None

    try:
        openai.api_key = openai_key
        # Menggunakan Chat Completions API
        response = openai.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Anda adalah asisten AI yang ahli dalam menganalisis dokumen hukum dan kontrak. Berikan jawaban yang komprehensif, relevan, dan mudah dipahami."},
                {"role": "user", "content": f"{prompt_template}\n\nKontrak:\n{contract_text}"}
            ],
            temperature=temp,
            max_tokens=tokens
        )
        return response.choices[0].message.content
    except openai.AuthenticationError:
        st.error("🚫 Autentikasi Gagal: API Key Anda salah atau tidak valid. Silakan periksa kembali.")
        return None
    except openai.RateLimitError:
        st.error("⏳ Batas Frekuensi Tercapai: Anda telah mencapai batas frekuensi API OpenAI. Coba lagi sebentar lagi.")
        return None
    except openai.APITimeoutError:
        st.error("⏰ Timeout API: Permintaan ke OpenAI memakan waktu terlalu lama. Coba lagi.")
        return None
    except openai.APIConnectionError as e:
        st.error(f"🔌 Kesalahan Koneksi API: Tidak dapat terhubung ke OpenAI. Periksa koneksi internet Anda. Detail: {e}")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan tak terduga saat berkomunikasi dengan OpenAI: {e}")
        return None

# --- Bagian Utama Aplikasi ---
uploaded_file = st.file_uploader("⬆️ Unggah dokumen kontrak PDF Anda", type="pdf")

if uploaded_file is not None:
    st.success(f"✅ File '{uploaded_file.name}' berhasil diunggah!")

    # Baca teks dari PDF
    with st.spinner("⏳ Mengekstrak teks dari PDF..."):
        contract_text = extract_text_from_pdf(uploaded_file.getvalue())

    if not contract_text.strip():
        st.warning("⚠️ Tidak dapat mengekstrak teks yang berarti dari PDF ini. Pastikan PDF tidak kosong atau berbasis gambar (scan).")
    else:
        st.subheader("📜 Teks yang Diekstrak dari Kontrak:")
        # Tampilkan sebagian teks dengan expander
        with st.expander("Lihat teks lengkap (klik untuk memperluas)"):
            st.text(contract_text[:1500] + "..." if len(contract_text) > 1500 else contract_text)

        # Informasi token
        num_tokens = count_tokens(contract_text, model_choice)
        st.info(f"💡 Perkiraan jumlah token teks kontrak: **{num_tokens}** (Model: {model_choice})")
        st.warning("Perhatikan batas token model. Teks yang terlalu panjang mungkin terpotong atau menimbulkan biaya lebih.")

        st.subheader("🧠 Pilih Jenis Analisis atau Buat Kustom:")

        analysis_options = {
            "Ringkasan Kontrak": "Mohon berikan ringkasan singkat dari poin-poin penting, tujuan, dan pihak-pihak dalam kontrak ini.",
            "Identifikasi Pihak & Peran": "Siapa saja pihak-pihak yang terlibat dalam kontrak ini? Jelaskan peran dan identitas masing-masing.",
            "Kewajiban & Hak Utama": "Jelaskan secara detail kewajiban dan hak utama masing-masing pihak berdasarkan kontrak ini.",
            "Ketentuan Pembayaran": "Analisis semua klausul terkait pembayaran, termasuk jadwal, jumlah, mata uang, dan konsekuensi keterlambatan.",
            "Jangka Waktu & Perpanjangan": "Berapa lama jangka waktu kontrak ini? Apakah ada klausul perpanjangan otomatis atau manual?",
            "Klausul Terminasi & Force Majeure": "Jelaskan kondisi-kondisi di mana kontrak dapat diakhiri (terminasi) dan apa yang diatur dalam klausul 'Force Majeure'.",
            "Analisis Risiko & Potensi Masalah": "Identifikasi potensi risiko, klausul yang ambigu, atau poin-poin yang dapat menimbulkan perselisihan atau kerugian bagi salah satu pihak.",
            "Bahasa Hukum Disederhanakan": "Jelaskan kontrak ini dalam bahasa yang sederhana dan mudah dipahami oleh non-hukum, fokus pada implikasi praktis.",
            "Kustom": "Masukkan perintah analisis Anda sendiri"
        }

        selected_analysis_type = st.selectbox(
            "Pilih jenis analisis cepat:",
            list(analysis_options.keys())
        )

        prompt_input = ""
        if selected_analysis_type == "Kustom":
            prompt_input = st.text_area(
                "Masukkan perintah analisis kustom Anda:",
                "Jelaskan secara detail tentang hak dan kewajiban masing-masing pihak, dan identifikasi potensi risiko yang mungkin timbul.",
                height=150,
                placeholder="Misalnya: 'Identifikasi semua tanggal penting dan tenggat waktu dalam kontrak ini.'"
            )
        else:
            prompt_input = analysis_options[selected_analysis_type]

        if st.button("🚀 Mulai Analisis Kontrak", type="primary"):
            if not openai_api_key:
                st.warning("⚠️ Harap masukkan OpenAI API Key Anda di sidebar sebelum memulai analisis.")
            elif not contract_text.strip():
                st.warning("⚠️ Tidak ada teks kontrak yang dapat dianalisis.")
            elif not prompt_input.strip():
                st.warning("⚠️ Harap masukkan perintah analisis.")
            else:
                with st.spinner(f"⏳ Menganalisis kontrak menggunakan {model_choice}..."):
                    analysis_result = analyze_contract_with_openai(
                        contract_text,
                        openai_api_key,
                        model_choice,
                        prompt_input,
                        temperature,
                        max_tokens
                    )
                    if analysis_result:
                        st.subheader(f"✨ Hasil Analisis: {selected_analysis_type}")
                        st.markdown(analysis_result) # Menggunakan markdown agar format lebih rapi
                        st.info("⚠️ **Penting:** Hasil analisis ini dihasilkan oleh AI dan harus diverifikasi secara independen oleh profesional hukum atau pihak yang berwenang sebelum digunakan untuk pengambilan keputusan penting.")
else:
    st.info("⬆️ Unggah file PDF kontrak Anda untuk memulai analisis.")

st.markdown("---")
st.markdown("Dibangun dengan ❤️ menggunakan Streamlit dan OpenAI API.")
