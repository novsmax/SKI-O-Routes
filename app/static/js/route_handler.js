

let startPointSelected = false;
let endPointSelected = false;
let startX = 0;
let startY = 0;
let endX = 0;
let endY = 0;

let startJunctionX = 0;
let startJunctionY = 0;
let endJunctionX = 0;
let endJunctionY = 0;
let junctionsFound = false;

let imageNaturalWidth = 0;
let imageNaturalHeight = 0;

let routeFound = false;
let routeImagePath = null;

let routeModal;
let errorModal;

let currentMapId;

document.addEventListener('DOMContentLoaded', function() {
    const pathParts = window.location.pathname.split('/');
    const mapIdIndex = pathParts.indexOf('maps') + 1;
    if (mapIdIndex > 0 && mapIdIndex < pathParts.length) {
        currentMapId = pathParts[mapIdIndex];
        console.log("Получен ID карты из URL:", currentMapId);
    } else {
        console.error("Не удалось получить ID карты из URL");
    }

    routeModal = new bootstrap.Modal(document.getElementById('routeModal'));
    errorModal = new bootstrap.Modal(document.getElementById('errorModal'));

    const routeMapImage = document.getElementById('routeMapImage');
    if (routeMapImage) {
        routeMapImage.addEventListener('click', mapClickHandler);
    } else {
        console.error("Не найден элемент routeMapImage");
    }

    initializeImageSizes();

    document.getElementById('routeModal').addEventListener('shown.bs.modal', function() {
        initializeImageSizes();
    });
});

function initializeImageSizes() {
    const routeMapImage = document.getElementById('routeMapImage');

    if (routeMapImage.complete) {
        updateImageSizes(routeMapImage);
    } else {
        routeMapImage.onload = function() {
            updateImageSizes(this);
        };
    }
}

function updateImageSizes(imgElement) {
    imageNaturalWidth = imgElement.naturalWidth;
    imageNaturalHeight = imgElement.naturalHeight;

    const displayedWidth = imgElement.width;
    const displayedHeight = imgElement.height;

    console.log('Размер оригинального изображения:', imageNaturalWidth, 'x', imageNaturalHeight);
    console.log('Размер отображаемого изображения:', displayedWidth, 'x', displayedHeight);
    console.log('Соотношение масштаба:', imageNaturalWidth / displayedWidth, imageNaturalHeight / displayedHeight);
}

async function analyzeMap() {
    if (!currentMapId) {
        console.error("ID карты не определен");
        showError("Не удалось определить ID карты");
        return;
    }


    const loadingOverlay = document.getElementById('loadingOverlay');
    loadingOverlay.style.display = 'flex';


    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Анализируем...';

    try {

        const response = await fetch(`/maps/${currentMapId}/analyze`, {
            method: 'POST'
        });

        const data = await response.json();


        loadingOverlay.style.display = 'none';

        if (data.success) {

            setTimeout(() => {
                window.location.reload();
            }, 1000);
        } else {

            showError(data.error || 'Произошла ошибка при анализе карты');


            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="bi bi-graph-up"></i> Анализировать карту';
        }
    } catch (error) {

        loadingOverlay.style.display = 'none';

        console.error('Ошибка:', error);
        showError('Произошла ошибка при анализе карты');


        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="bi bi-graph-up"></i> Анализировать карту';
    }
}


function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    errorModal.show();
}


function openRouteModal() {

    resetPoints();


    routeFound = false;
    routeImagePath = null;


    document.getElementById('routeMapContainer').style.display = 'block';
    document.getElementById('routeResultContainer').style.display = 'none';
    document.getElementById('routeInfoCard').style.display = 'none';


    routeModal.show();
}


function mapClickHandler(event) {
    if (routeFound) return;

    const mapElement = event.target;
    const startPoint = document.getElementById('modalStartPoint');
    const endPoint = document.getElementById('modalEndPoint');
    const startLabel = document.getElementById('startPointLabel');
    const endLabel = document.getElementById('endPointLabel');


    const rect = mapElement.getBoundingClientRect();


    const clientX = event.clientX;
    const clientY = event.clientY;


    const clickX = clientX - rect.left;
    const clickY = clientY - rect.top;

    console.log('Клик по карте:', clickX, clickY);


    if (!startPointSelected) {

        startX = clickX;
        startY = clickY;

        startPoint.style.left = `${clickX}px`;
        startPoint.style.top = `${clickY}px`;
        startPoint.style.display = 'block';

        startLabel.style.left = `${clickX}px`;
        startLabel.style.top = `${clickY}px`;
        startLabel.style.display = 'block';

        startPointSelected = true;

        if (endPointSelected) {
            findNearestJunctions();
        }
    } else if (!endPointSelected) {
        endX = clickX;
        endY = clickY;

        endPoint.style.left = `${clickX}px`;
        endPoint.style.top = `${clickY}px`;
        endPoint.style.display = 'block';

        endLabel.style.left = `${clickX}px`;
        endLabel.style.top = `${clickY}px`;
        endLabel.style.display = 'block';

        endPointSelected = true;

        findNearestJunctions();
    }
}

async function findNearestJunctions() {
    if (!startPointSelected || !endPointSelected || !currentMapId) {
        console.error("Не все точки выбраны или не определен ID карты");
        return;
    }

    const modalLoadingOverlay = document.getElementById('modalLoadingOverlay');
    modalLoadingOverlay.style.display = 'flex';

    try {
        const routeMapImage = document.getElementById('routeMapImage');
        const displayedWidth = routeMapImage.width;
        const displayedHeight = routeMapImage.height;

        const scaleRatioX = imageNaturalWidth / displayedWidth;
        const scaleRatioY = imageNaturalHeight / displayedHeight;

        const originalStartX = Math.round(startX * scaleRatioX);
        const originalStartY = Math.round(startY * scaleRatioY);
        const originalEndX = Math.round(endX * scaleRatioX);
        const originalEndY = Math.round(endY * scaleRatioY);

        const formData = new FormData();
        formData.append('start_x', originalStartX);
        formData.append('start_y', originalStartY);
        formData.append('end_x', originalEndX);
        formData.append('end_y', originalEndY);

        console.log('Координаты клика:', startX, startY, endX, endY);
        console.log('Соотношение масштаба:', scaleRatioX, scaleRatioY);
        console.log('Отправляем координаты (в исходном масштабе):', originalStartX, originalStartY, originalEndX, originalEndY);

        const response = await fetch(`/maps/${currentMapId}/find_junctions`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        console.log('Получены перекрестки:', data);

        modalLoadingOverlay.style.display = 'none';

        if (data.success) {
            const startJunction = document.getElementById('startJunction');
            const endJunction = document.getElementById('endJunction');
            const startConnectionLine = document.getElementById('startConnectionLine');
            const endConnectionLine = document.getElementById('endConnectionLine');

            const originalStartJunctionX = data.start_junction.x;
            const originalStartJunctionY = data.start_junction.y;
            const originalEndJunctionX = data.end_junction.x;
            const originalEndJunctionY = data.end_junction.y;

            startJunctionX = originalStartJunctionX / scaleRatioX;
            startJunctionY = originalStartJunctionY / scaleRatioY;
            endJunctionX = originalEndJunctionX / scaleRatioX;
            endJunctionY = originalEndJunctionY / scaleRatioY;

            console.log('Оригинальные координаты перекрестков:', originalStartJunctionX, originalStartJunctionY, originalEndJunctionX, originalEndJunctionY);
            console.log('Пересчитанные координаты перекрестков:', startJunctionX, startJunctionY, endJunctionX, endJunctionY);

            startJunction.style.left = `${startJunctionX}px`;
            startJunction.style.top = `${startJunctionY}px`;
            startJunction.style.display = 'block';

            endJunction.style.left = `${endJunctionX}px`;
            endJunction.style.top = `${endJunctionY}px`;
            endJunction.style.display = 'block';

            drawConnectionLine(startConnectionLine, startX, startY, startJunctionX, startJunctionY);
            drawConnectionLine(endConnectionLine, endX, endY, endJunctionX, endJunctionY);

            junctionsFound = true;

            document.getElementById('findRouteBtn').disabled = false;
        } else {
            showError(data.error || 'Не удалось найти ближайшие перекрестки');
        }
    } catch (error) {
        modalLoadingOverlay.style.display = 'none';

        console.error('Ошибка:', error);
        showError('Произошла ошибка при поиске ближайших перекрестков');
    }
}

function drawConnectionLine(lineElement, x1, y1, x2, y2) {
    const length = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));
    const angle = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;

    lineElement.style.left = `${x1}px`;
    lineElement.style.top = `${y1}px`;
    lineElement.style.width = `${length}px`;
    lineElement.style.transform = `rotate(${angle}deg)`;

    lineElement.classList.add('animated-dashed');

    lineElement.style.display = 'block';
}

function resetPoints() {
    startPointSelected = false;
    endPointSelected = false;
    junctionsFound = false;

    document.getElementById('modalStartPoint').style.display = 'none';
    document.getElementById('modalEndPoint').style.display = 'none';
    document.getElementById('startPointLabel').style.display = 'none';
    document.getElementById('endPointLabel').style.display = 'none';
    document.getElementById('startJunction').style.display = 'none';
    document.getElementById('endJunction').style.display = 'none';
    document.getElementById('startConnectionLine').style.display = 'none';
    document.getElementById('endConnectionLine').style.display = 'none';

    document.getElementById('findRouteBtn').disabled = true;
}


async function findRoute() {
    if (!startPointSelected || !endPointSelected || !junctionsFound || !currentMapId) {
        showError('Необходимо выбрать как начальную, так и конечную точку маршрута');
        return;
    }

    const modalLoadingOverlay = document.getElementById('modalLoadingOverlay');
    modalLoadingOverlay.style.display = 'flex';

    const findRouteBtn = document.getElementById('findRouteBtn');
    findRouteBtn.disabled = true;
    findRouteBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Строим маршрут...';

    try {
        const routeMapImage = document.getElementById('routeMapImage');
        const displayedWidth = routeMapImage.width;
        const displayedHeight = routeMapImage.height;

        const scaleRatioX = imageNaturalWidth / displayedWidth;
        const scaleRatioY = imageNaturalHeight / displayedHeight;

        const originalStartJunctionX = Math.round(startJunctionX * scaleRatioX);
        const originalStartJunctionY = Math.round(startJunctionY * scaleRatioY);
        const originalEndJunctionX = Math.round(endJunctionX * scaleRatioX);
        const originalEndJunctionY = Math.round(endJunctionY * scaleRatioY);

        const formData = new FormData();
        formData.append('start_x', originalStartJunctionX);
        formData.append('start_y', originalStartJunctionY);
        formData.append('end_x', originalEndJunctionX);
        formData.append('end_y', originalEndJunctionY);

        console.log('Отправляем координаты перекрестков для маршрута (в исходном масштабе):',
                    originalStartJunctionX, originalStartJunctionY,
                    originalEndJunctionX, originalEndJunctionY);

        const response = await fetch(`/maps/${currentMapId}/find_route`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        modalLoadingOverlay.style.display = 'none';

        if (data.success) {
            const cacheParam = new Date().getTime();
            routeImagePath = data.route_path + '?t=' + cacheParam;

            showRouteResult(routeImagePath);
            routeFound = true;

            findRouteBtn.disabled = false;
            findRouteBtn.innerHTML = '<i class="bi bi-signpost-2"></i> Построить маршрут';
        } else {
            showError(data.error || 'Не удалось построить маршрут');

            findRouteBtn.disabled = false;
            findRouteBtn.innerHTML = '<i class="bi bi-signpost-2"></i> Построить маршрут';
        }
    } catch (error) {
        modalLoadingOverlay.style.display = 'none';

        console.error('Ошибка:', error);
        showError('Произошла ошибка при построении маршрута');

        findRouteBtn.disabled = false;
        findRouteBtn.innerHTML = '<i class="bi bi-signpost-2"></i> Построить маршрут';
    }
}


function showRouteResult(imagePath) {
    document.getElementById('routeMapContainer').style.display = 'none';

    const resultContainer = document.getElementById('routeResultContainer');
    resultContainer.style.display = 'block';

    const resultImage = document.getElementById('routeResultImage');
    resultImage.src = imagePath;

    const routeInfoCard = document.getElementById('routeInfoCard');
    routeInfoCard.style.display = 'block';

    const routeInfoContent = document.getElementById('routeInfoContent');
    routeInfoContent.innerHTML = `
        <p><strong>Маршрут успешно построен!</strong></p>
        <p>Оптимальный маршрут между выбранными точками отображается на карте.</p>
    `;
}

function backToMapSelection() {
    routeFound = false;
    document.getElementById('routeMapContainer').style.display = 'block';
    document.getElementById('routeResultContainer').style.display = 'none';
    document.getElementById('routeInfoCard').style.display = 'none';

    resetPoints();
}