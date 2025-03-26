<?php
require_once __DIR__ . '/../config.php';

function fetchWeatherData($endpoint, $params = []) {
    $url = BASE_URL . $endpoint . '?' . http_build_query(array_merge(['key' => API_KEY], $params));

    $curl = curl_init();
    curl_setopt($curl, CURLOPT_URL, $url);
    curl_setopt($curl, CURLOPT_RETURNTRANSFER, true);
    
    // Pour le débogage
    error_log("Appel API : " . $url);
    
    $response = curl_exec($curl);
    
    // Vérifier s'il y a des erreurs
    if ($response === false) {
        error_log('Erreur cURL: ' . curl_error($curl));
    }
    
    curl_close($curl);
    
    $result = json_decode($response, true);
    
    // Pour le débogage
    if (json_last_error() !== JSON_ERROR_NONE) {
        error_log('Erreur JSON: ' . json_last_error_msg());
    }
    
    return $result;
}

// Fonction pour récupérer les prévisions sur 7 jours - ASSUREZ-VOUS QUE CETTE FONCTION EST PRÉSENTE
function fetchWeeklyForecast($city) {
    $result = fetchWeatherData('forecast.json', ['q' => $city, 'days' => 7]);
    
    // Pour le débogage
    if (!isset($result['forecast'])) {
        error_log('Réponse API pour ' . $city . ': ' . json_encode($result));
    }
    
    return $result;
}
?>