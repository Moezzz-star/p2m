// Aperçu de l’image avant envoi
const fileInput  = document.getElementById('fileInput');
const previewImg = document.getElementById('preview');

if (fileInput) {
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (!file) { return; }

        const reader = new FileReader();
        reader.onload = e => {
            previewImg.src = e.target.result;
            previewImg.classList.remove('d-none');
        };
        reader.readAsDataURL(file);
    });
}
