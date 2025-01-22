<?php
require_once __DIR__ . '/../config.php';


function fetchWeatherData($endpoint, $params = []) {

    $url = BASE_URL . $endpoint . '?' . http_build_query(array_merge(['key' => API_KEY], $params));

    $curl = curl_init();
    curl_setopt($curl, CURLOPT_URL, $url);
    curl_setopt($curl, CURLOPT_RETURNTRANSFER, true);

    $response = curl_exec($curl);
    curl_close($curl);

    return json_decode($response, true);
}
