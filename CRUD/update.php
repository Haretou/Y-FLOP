<?php
require_once __DIR__ . '/../config.php';  // Inclure la connexion à la base de données

// Obtenir la connexion PDO
$pdo = getDatabaseConnection();

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST['id'], $_POST['temperature'])) {
    $id = $_POST['id'];
    $temperature = $_POST['temperature'];

    if ($pdo) {
        // Préparer la requête de mise à jour
        $query = "UPDATE searches SET temperature = :temperature WHERE id = :id";
        $stmt = $pdo->prepare($query);

        // Lier les paramètres
        $stmt->bindParam(':temperature', $temperature);
        $stmt->bindParam(':id', $id);

        // Exécuter la requête
        if ($stmt->execute()) {
            echo "Recherche mise à jour avec succès!";
        } else {
            echo "Erreur lors de la mise à jour de la recherche.";
        }
    } else {
        echo "Erreur de connexion à la base de données!";
    }
}
?>

<form method="POST" action="">
    <label for="id">ID de la Recherche:</label>
    <input type="number" id="id" name="id" required><br>

    <label for="temperature">Nouvelle Température:</label>
    <input type="number" id="temperature" name="temperature" required><br>

    <button type="submit">Mettre à jour la recherche</button>
</form>
