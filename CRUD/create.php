<?php
require_once __DIR__ . '/../config.php';  // Inclure la connexion à la base de données

// Obtenir la connexion PDO
$pdo = getDatabaseConnection();

function saveSearch($city, $temperature, $condition) {
    // Code pour enregistrer les données dans la base de données
    // Par exemple :
    $db = new PDO('sqlite:database.db'); // Adapter selon ton environnement
    $stmt = $db->prepare("INSERT INTO searches (city, temperature, condition) VALUES (?, ?, ?)");
    $stmt->execute([$city, $temperature, $condition]);}

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST['city'], $_POST['temperature'])) {
    $city = $_POST['city'];
    $temperature = $_POST['temperature'];

    if ($pdo) {
        // Préparer la requête d'insertion
        $query = "INSERT INTO searches (city, temperature) VALUES (:city, :temperature)";
        $stmt = $pdo->prepare($query);

        // Lier les paramètres
        $stmt->bindParam(':city', $city);
        $stmt->bindParam(':temperature', $temperature);

        // Exécuter la requête
        if ($stmt->execute()) {
            echo "Recherche enregistrée avec succès!";
        } else {
            echo "Erreur lors de l'enregistrement de la recherche.";
        }
    } else {
        echo "Erreur de connexion à la base de données!";
    }
}
?>

<form method="POST" action="">
    <label for="city">Ville:</label>
    <input type="text" id="city" name="city" required><br>

    <label for="temperature">Température:</label>
    <input type="number" id="temperature" name="temperature" required><br>

    <button type="submit">Enregistrer la recherche</button>
</form>
