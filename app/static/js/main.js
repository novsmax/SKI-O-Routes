/**
 * Основной JavaScript файл для цифрового ассистента спортивного ориентирования
 */

document.addEventListener('DOMContentLoaded', function() {
    // Инициализация всплывающих подсказок
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Валидация формы загрузки файла
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const fileName = this.files[0]?.name;
            if (fileName) {
                const fileExtension = fileName.split('.').pop().toLowerCase();
                const allowedExtensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp'];

                if (!allowedExtensions.includes(fileExtension)) {
                    alert('Пожалуйста, выберите изображение в формате JPG, PNG или GIF');
                    this.value = '';
                }
            }
        });
    }

    // Предварительный просмотр загружаемого файла
    const filePreview = document.getElementById('filePreview');
    if (fileInput && filePreview) {
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    filePreview.src = e.target.result;
                    filePreview.style.display = 'block';
                };
                reader.readAsDataURL(this.files[0]);
            }
        });
    }
});

/**
 * Функция для форматирования даты в локальном формате
 * @param {string} dateString - Строка с датой
 * @param {string} locale - Локаль (по умолчанию 'ru-RU')
 * @returns {string} Отформатированная дата
 */
function formatDate(dateString, locale = 'ru-RU') {
    const date = new Date(dateString);
    return date.toLocaleDateString(locale, {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric'
    });
}