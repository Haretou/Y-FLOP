document.addEventListener('DOMContentLoaded', function() {
    // Définir la date actuelle
    setCurrentDate();
    
    // Tenter de charger une commune par défaut (la première de la liste)
    const cityItems = document.querySelectorAll('.city-item');
    if (cityItems.length > 0) {
        loadCommuneWeather(cityItems[0].textContent);
    } else {
        // Si aucune commune n'est disponible, charger les prévisions générales
        fetchForecast();
    }
    
    // Configurer la recherche de communes
    setupCommuneSearch();
    
    // Gérer le formulaire de prédiction
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            submitPredictionForm();
        });
    }
    
    // Simuler des données en temps réel pour la démo
    updateCurrentWeather();
});

// Fonction pour afficher le jour et la date actuels
function setCurrentDate() {
    const now = new Date();
    
    // Tableau des jours en français
    const days = [
        'Dimanche',
        'Lundi',
        'Mardi',
        'Mercredi',
        'Jeudi',
        'Vendredi',
        'Samedi'
    ];
    
    // Tableau des mois en français
    const months = [
        'janvier',
        'février',
        'mars',
        'avril',
        'mai',
        'juin',
        'juillet',
        'août',
        'septembre',
        'octobre',
        'novembre',
        'décembre'
    ];
    
    // Formatage du jour
    const dayName = days[now.getDay()];
    
    // Formatage de la date
    const day = now.getDate();
    const month = months[now.getMonth()];
    const year = now.getFullYear();
    
    // Affichage dans le HTML
    const currentDayElement = document.getElementById('current-day');
    const currentDateElement = document.getElementById('current-date');
    
    if (currentDayElement) {
        currentDayElement.textContent = dayName;
    }
    
    if (currentDateElement) {
        currentDateElement.textContent = `${day} ${month} ${year}`;
    }
}

function setupCommuneSearch() {
    const searchInput = document.getElementById('city-search');
    const searchResults = document.getElementById('search-results');
    
    if (!searchInput || !searchResults) return;
    
    searchInput.addEventListener('input', async function() {
        const query = this.value.trim();
        
        if (query.length < 2) {
            searchResults.innerHTML = '';
            searchResults.style.display = 'none';
            return;
        }
        
        try {
            const response = await fetch(`/search_city?q=${encodeURIComponent(query)}`);
            const communes = await response.json();
            
            displaySearchResults(communes, searchResults);
        } catch (error) {
            console.error('Erreur lors de la recherche des communes:', error);
        }
    });
    
    // Cacher les résultats quand on clique ailleurs
    document.addEventListener('click', function(e) {
        if (e.target !== searchInput && e.target !== searchResults) {
            searchResults.style.display = 'none';
        }
    });
}

function displaySearchResults(communes, resultsContainer) {
    resultsContainer.innerHTML = '';
    
    if (communes.length === 0) {
        resultsContainer.innerHTML = '<div class="no-results">Aucune commune trouvée</div>';
    } else {
        communes.forEach(commune => {
            const resultItem = document.createElement('div');
            resultItem.className = 'search-result-item';
            resultItem.textContent = `${commune.commune} (${commune.code_commune})`;
            resultItem.addEventListener('click', function() {
                loadCommuneWeather(commune.commune);
                document.getElementById('city-search').value = commune.commune;
                resultsContainer.style.display = 'none';
            });
            
            resultsContainer.appendChild(resultItem);
        });
    }
    
    resultsContainer.style.display = 'block';
}

async function loadCommuneWeather(communeName) {
    try {
        // Mettre à jour la date lors du chargement d'une nouvelle commune
        setCurrentDate();
        
        // Charger les informations météo actuelles de la commune
        const weatherResponse = await fetch(`/city_weather/${encodeURIComponent(communeName)}`);
        const weatherData = await weatherResponse.json();
        
        // Mettre à jour l'affichage avec les données météo
        document.getElementById('city-name').textContent = communeName;
        document.getElementById('current-temp').textContent = weatherData.temperature;
        document.getElementById('current-humidity').textContent = weatherData.humidity;
        document.getElementById('current-precipitation').textContent = weatherData.precipitation;
        document.getElementById('current-wind').textContent = weatherData.wind_speed;
        
        // Charger les prévisions pour cette commune
        const forecastResponse = await fetch(`/forecast/${encodeURIComponent(communeName)}`);
        const forecastData = await forecastResponse.json();
        
        // Mettre à jour l'affichage des prévisions
        document.getElementById('forecast-city-name').textContent = communeName;
        displayForecast(forecastData);
        createTemperatureChart(forecastData);
    } catch (error) {
        console.error(`Erreur lors du chargement des données pour ${communeName}:`, error);
    }
}

async function fetchForecast() {
    try {
        const response = await fetch('/forecast');
        const data = await response.json();
        
        // Afficher les prévisions
        displayForecast(data);
        
        // Créer le graphique
        createTemperatureChart(data);
    } catch (error) {
        console.error('Erreur lors de la récupération des prévisions:', error);
    }
}

function displayForecast(forecasts) {
    const container = document.getElementById('forecast-container');
    container.innerHTML = '';
    
    forecasts.forEach(day => {
        const card = document.createElement('div');
        card.className = 'forecast-card';
        
        const date = new Date(day.date);
        const options = { weekday: 'short', day: 'numeric' };
        
        card.innerHTML = `
            <div class="date">${date.toLocaleDateString('fr-FR', options)}</div>
            <div class="temp">${day.temperature}°C</div>
            <div class="humidity">Hum: ${day.humidity}%</div>
            <div class="precipitation">Prec: ${day.precipitation} mm</div>
        `;
        
        container.appendChild(card);
    });
}

function createTemperatureChart(forecasts) {
    const ctx = document.getElementById('temperature-chart').getContext('2d');
    
    // Détruire le graphique existant s'il y en a un
    if (window.temperatureChart) {
        window.temperatureChart.destroy();
    }
    
    const labels = forecasts.map(day => {
        const date = new Date(day.date);
        const options = { weekday: 'short', day: 'numeric' };
        return date.toLocaleDateString('fr-FR', options);
    });
    
    const temperatures = forecasts.map(day => day.temperature);
    
    window.temperatureChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Température (°C)',
                data: temperatures,
                backgroundColor: 'rgba(52, 152, 219, 0.2)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 2,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

async function submitPredictionForm() {
    const form = document.getElementById('prediction-form');
    const formData = new FormData(form);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        // Afficher le résultat
        document.getElementById('predicted-temp').textContent = data.temperature;
        document.getElementById('prediction-result').style.display = 'block';
    } catch (error) {
        console.error('Erreur lors de la prédiction:', error);
        alert('Erreur lors de la prédiction. Veuillez réessayer.');
    }
}

function updateCurrentWeather() {
    // Vérifier si les éléments existent avant de mettre à jour
    const currentTemp = document.getElementById('current-temp');
    const currentHumidity = document.getElementById('current-humidity');
    const currentWind = document.getElementById('current-wind');
    
    // Pour la démo, on simule des données météo actuelles
    if (currentTemp) currentTemp.textContent = (20 + Math.random() * 5).toFixed(1);
    if (currentHumidity) currentHumidity.textContent = (60 + Math.random() * 20).toFixed(0);
    if (currentWind) currentWind.textContent = (10 + Math.random() * 15).toFixed(1);
}