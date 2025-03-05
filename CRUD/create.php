<?php
require_once __DIR__ . '/../config.php';  // Inclure la connexion à la base de données

// Obtenir la connexion PDO
$pdo = getDatabaseConnection();

// Vérifier si la connexion à la base de données a échoué
if ($pdo === null) {
    die("Erreur de connexion à la base de données!");
}

function saveSearch($city, $temperature, $condition) {
    global $pdo;  // Utiliser la connexion PDO globale

    try {
        // Préparer la requête d'insertion avec tous les paramètres
        $query = "INSERT INTO searches (city, temperature, `condition`) VALUES (:city, :temperature, :condition)";
        $stmt = $pdo->prepare($query);

        // Lier les paramètres
        $stmt->bindParam(':city', $city);
        $stmt->bindParam(':temperature', $temperature);
        $stmt->bindParam(':condition', $condition);

        // Exécuter la requête
       
    } catch (PDOException $e) {
        // Afficher l'erreur en cas d'échec de la requête
        echo "Erreur de base de données : " . $e->getMessage();
    }
}

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST['city'], $_POST['temperature'])) {
    $city = $_POST['city'];
    $temperature = $_POST['temperature'];
    $condition = $_POST['condition'] ?? '';  // Condition est optionnelle, peut être vide si non fournie

    saveSearch($city, $temperature, $condition);  // Appel de la fonction avec les bons paramètres
}

?>


