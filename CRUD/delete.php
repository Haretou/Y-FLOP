<?php
require_once __DIR__ . '/../config.php';

function deleteSearch($id) {
    $conn = getDatabaseConnection();
    $stmt = $conn->prepare('DELETE FROM searches WHERE id = :id');
    $stmt->execute(['id' => $id]);
}
