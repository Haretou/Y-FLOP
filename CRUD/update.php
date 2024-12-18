<?php
require_once __DIR__ . '/../config.php';

function updateSearch($id, $newCity) {
    $conn = getDatabaseConnection();
    $stmt = $conn->prepare('UPDATE searches SET city = :city WHERE id = :id');
    $stmt->execute(['city' => $newCity, 'id' => $id]);
}
