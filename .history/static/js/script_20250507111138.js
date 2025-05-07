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
    
    // S'assurer que la div de résultats est initialement masquée avec le bon style
    searchResults.style.display = 'none';
    
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
            // Afficher un message d'erreur à l'utilisateur
            searchResults.innerHTML = '<div class="no-results">Erreur de recherche</div>';
            searchResults.style.display = 'block';
        }
    });
    
    // Amélioration de la détection de clic en dehors
    document.addEventListener('click', function(e) {
        // Vérifier si le clic est en dehors de la zone de recherche
        if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.style.display = 'none';
        }
    });
    
    // Empêcher la propagation du clic à l'intérieur des résultats pour éviter qu'ils ne soient cachés
    searchResults.addEventListener('click', function(e) {
        e.stopPropagation();
    });
}

function displaySearchResults(communes, resultsContainer) {
    // Vider le conteneur
    resultsContainer.innerHTML = '';
    
    if (communes.length === 0) {
        const noResults = document.createElement('div');
        noResults.className = 'no-results';
        noResults.textContent = 'Aucune commune trouvée';
        resultsContainer.appendChild(noResults);
    } else {
        communes.forEach(commune => {
            const resultItem = document.createElement('div');
            resultItem.className = 'search-result-item';
            resultItem.textContent = commune.commune ? `${commune.commune} (${commune.code_commune || ''})` : commune;
            
            resultItem.addEventListener('click', function() {
                const communeName = commune.commune || commune;
                loadCommuneWeather(communeName);
                document.getElementById('city-search').value = communeName;
                resultsContainer.style.display = 'none';
            });
            
            resultsContainer.appendChild(resultItem);
        });
    }
    
    // S'assurer que le conteneur est visible avec la bonne position
    resultsContainer.style.display = 'block';
    
    // Ajouter une classe active pour améliorer la visibilité
    resultsContainer.classList.add('active');
}

async function loadCommuneWeather(communeName) {
    try {
        // Mettre à jour la date lors du chargement d'une nouvelle commune
        setCurrentDate();
        
        // Montrer un état de chargement
        const sections = document.querySelectorAll('.current-weather, .forecast');
        sections.forEach(section => section.classList.add('loading'));
        
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
        
        // Mettre en surbrillance la ville active dans la liste
        const cityItems = document.querySelectorAll('.city-item');
        cityItems.forEach(item => {
            if (item.textContent.trim() === communeName.trim()) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
        
        // Retirer l'état de chargement
        sections.forEach(section => section.classList.remove('loading'));
    } catch (error) {
        console.error(`Erreur lors du chargement des données pour ${communeName}:`, error);
        // Afficher un message d'erreur à l'utilisateur
        sections.forEach(section => section.classList.remove('loading'));
    }
}