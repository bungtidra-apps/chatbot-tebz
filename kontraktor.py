import streamlit as st
import openai
import PyPDF2
import io
import tiktoken # Untuk menghitung token, opsional

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Penganalisis Kontrak PDF dengan OpenAI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Penganalisis Kontrak PDF dengan OpenAI")
st.write("Unggah dokumen kontrak PDF Anda dan dapatkan analisis dari AI.")

# --- Sidebar untuk API Key ---
with st.sidebar:
    st.header("Konfigurasi API Key")
    openai_api_key = st.text_input(
        "Masukkan OpenAI API Key Anda",
        type="password",
        help="Anda bisa mendapatkan API key dari https://platform.openai.com/account/api-keys"
    )
    st.info("API Key Anda tidak akan disimpan.")

# --- Fungsi untuk Ekstraksi Teks dari PDF ---
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text() or "" # Handle None jika tidak ada teks
    return text

# --- Fungsi untuk Menghitung Token (opsional) ---
def count_tokens(text, model_name="gpt-4"):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except KeyError:
        return len(text.split()) # Fallback jika model tidak dikenal tiktoken

# --- Fungsi untuk Analisis Kontrak dengan OpenAI ---
def analyze_contract_with_openai(contract_text, openai_key, prompt_template):
    if not openai_key:
        st.error("Silakan masukkan OpenAI API Key Anda di sidebar.")
        return None

    try:
        openai.api_key = openai_key
        # Menyesuaikan prompt untuk model yang lebih baru (chat completion)
        response = openai.chat.completions.create(
            model="gpt-4", # Anda bisa mengganti dengan "gpt-3.5-turbo" untuk biaya lebih rendah
            messages=[
                {"role": "system", "content": "Anda adalah asisten AI yang ahli dalam menganalisis dokumen hukum dan kontrak."},
                {"role": "user", "content": f"{prompt_template}\n\nKontrak:\n{contract_text}"}
            ],
            temperature=0.7,
            max_tokens=1500 # Sesuaikan sesuai kebutuhan
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Terjadi kesalahan saat berkomunikasi dengan OpenAI: {e}")
        return None

# --- Bagian Utama Aplikasi ---
uploaded_file = st.file_uploader("Unggah dokumen kontrak PDF Anda", type="pdf")

if uploaded_file is not None:
    st.success("File PDF berhasil diunggah!")

    # Baca teks dari PDF
    with st.spinner("Mengekstrak teks dari PDF..."):
        contract_text = extract_text_from_pdf(io.BytesIO(uploaded_file.getvalue()))
        if not contract_text.strip():
            st.warning("Tidak dapat mengekstrak teks dari PDF. Pastikan PDF dapat dibaca secara optik atau tidak kosong.")
        else:
            st.subheader("Teks yang Diekstrak dari Kontrak:")
            with st.expander("Lihat teks lengkap"):
                st.text(contract_text[:1000] + "..." if len(contract_text) > 1000 else contract_text)
            st.info(f"Jumlah perkiraan token teks kontrak: {count_tokens(contract_text)}")

            st.subheader("Pilih Jenis Analisis atau Buat Kustom:")

            analysis_options = {
                "Ringkasan Kontrak": "Mohon berikan ringkasan singkat dari poin-poin penting dalam kontrak ini.",
                "Identifikasi Pihak": "Siapa saja pihak-pihak yang terlibat dalam kontrak ini?",
                "Kewajiban Utama": "Apa saja kewajiban utama masing-masing pihak berdasarkan kontrak ini?",
                "Kondisi Pembayaran": "Jelaskan syarat dan ketentuan pembayaran dalam kontrak ini.",
                "Jangka Waktu Kontrak": "Berapa lama jangka waktu kontrak ini?",
                "Klausul Terminasi": "Jelaskan klausul terminasi atau pengakhiran kontrak ini.",
                "Analisis Risiko": "Identifikasi potensi risiko atau klausul yang merugikan salah satu pihak.",
                "Kustom": "Masukkan perintah analisis Anda sendiri"
            }

            selected_analysis = st.selectbox(
                "Pilih jenis analisis cepat:",
                list(analysis_options.keys())
            )

            prompt_input = ""
            if selected_analysis == "Kustom":
                prompt_input = st.text_area(
                    "Masukkan perintah analisis kustom Anda:",
                    "Jelaskan secara detail tentang hak dan kewajiban masing-masing pihak.",
                    height=150
                )
            else:
                prompt_input = analysis_options[selected_analysis]

            if st.button("Mulai Analisis Kontrak"):
                if not openai_api_key:
                    st.warning("Harap masukkan OpenAI API Key Anda di sidebar sebelum memulai analisis.")
                elif not contract_text.strip():
                    st.warning("Tidak ada teks kontrak yang dapat dianalisis.")
                elif not prompt_input.strip():
                    st.warning("Harap masukkan perintah analisis.")
                else:
                    with st.spinner("Menganalisis kontrak dengan OpenAI..."):
                        full_prompt = prompt_input
                        analysis_result = analyze_contract_with_openai(
                            contract_text,
                            openai_api_key,
                            full_prompt
                        )
                        if analysis_result:
                            st.subheader(f"Hasil Analisis: {selected_analysis}")
                            st.write(analysis_result)
                            st.info("Penjelasan ini dihasilkan oleh AI dan harus diverifikasi oleh profesional hukum.")
