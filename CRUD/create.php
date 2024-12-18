<?php
require_once __DIR__ . '/../config.php';

function saveSearch($city, $temperature, $description) {
    $conn = getDatabaseConnection();
    $stmt = $conn->prepare('INSERT INTO searches (city, temperature, description) VALUES (:city, :temperature, :description)');
    $stmt->execute([
        'city' => $city,
        'temperature' => $temperature,
        'description' => $description
    ]);
}
