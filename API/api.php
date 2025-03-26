<?php
require_once __DIR__ . '/../config.php';

function fetchWeatherData($endpoint, $params = []) {
    $url = BASE_URL . $endpoint . '?' . http_build_query(array_merge(['key' => API_KEY], $params));

    $curl = curl_init();
    curl_setopt($curl, CURLOPT_URL, $url);
    curl_setopt($curl, CURLOPT_RETURNTRANSFER, true);
    
    $response = curl_exec($curl);
    
    if ($response === false) {
        error_log('Erreur cURL: ' . curl_error($curl));
    }
    
    curl_close($curl);
    
    $result = json_decode($response, true);
    
    return $result;
}

// Fonction pour récupérer les prévisions sur 7 jours
function fetchWeeklyForecast($city) {
    // Spécifier explicitement 7 jours dans la requête API
    return fetchWeatherData('forecast.json', ['q' => $city, 'days' => 7]);
}
?>