<?php
require_once __DIR__ . '/../config.php';


function getRecentSearches() {
    $conn = getDatabaseConnection();
    $stmt = $conn->query('SELECT city, temperature, description FROM searches ORDER BY date DESC LIMIT 5');
    return $stmt->fetchAll(PDO::FETCH_ASSOC);
}
