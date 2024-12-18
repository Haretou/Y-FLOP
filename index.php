<?php
require_once 'config.php'; // Assurez-vous que le fichier de configuration est inclus

// Fonction pour obtenir la connexion PDO
function getDatabaseConnection() {
    try {
        // Remplacez les informations suivantes avec les vôtres
        $pdo = new PDO('mysql:host=localhost;dbname=meteo_db', 'root', ''); // Changer les paramètres selon votre base de données
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        return $pdo;
    } catch (PDOException $e) {
        echo 'Erreur de connexion à la base de données: ' . $e->getMessage();
        return null;
    }
}

// Fonction pour enregistrer une recherche dans la base de données
function saveSearch($city, $temperature) {
    // Obtenir la connexion PDO
    $pdo = getDatabaseConnection();

    if ($pdo) {
        // Préparer la requête d'insertion
        $query = "INSERT INTO searches (city, temperature) VALUES (:city, :temperature)";
        $stmt = $pdo->prepare($query);

        // Lier les paramètres
        $stmt->bindParam(':city', $city);
        $stmt->bindParam(':temperature', $temperature);

        // Exécuter la requête et vérifier si l'insertion a réussi
        if ($stmt->execute()) {
            echo "Recherche enregistrée avec succès!";
        } else {
            echo "Erreur lors de l'enregistrement de la recherche.";
        }
    } else {
        echo "Erreur de connexion à la base de données!";
    }
}

// Fonction pour obtenir les recherches récentes
function getRecentSearches() {
    $pdo = getDatabaseConnection();
    if ($pdo) {
        // Requête pour récupérer les recherches récentes
        $query = "SELECT city, temperature, search_date FROM searches ORDER BY search_date DESC LIMIT 5";
        $stmt = $pdo->query($query);
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }
    return [];
}

// Si le formulaire est soumis, on appelle saveSearch pour enregistrer les données
if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_POST['city'], $_POST['temperature'])) {
    $city = $_POST['city'];
    $temperature = $_POST['temperature'];

    // Sauvegarder la recherche
    saveSearch($city, $temperature);
}
?>

<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Météo</title>
</head>
<body>
    <h1>Bienvenue sur l'API Météo</h1>

    <!-- Formulaire pour enregistrer une recherche -->
    <form method="POST" action="">
        <label for="city">Ville:</label>
        <input type="text" id="city" name="city" required><br>

        <label for="temperature">Température (°C):</label>
        <input type="number" step="0.1" id="temperature" name="temperature" required><br>

        <button type="submit">Enregistrer la recherche</button>
    </form>

    <h2>Recherches récentes</h2>
    <table border="1">
        <thead>
            <tr>
                <th>Ville</th>
                <th>Température (°C)</th>
                <th>Date de la recherche</th>
            </tr>
        </thead>
        <tbody>
            <?php
            // Afficher les recherches récentes
            $recentSearches = getRecentSearches();
            if (count($recentSearches) > 0) {
                foreach ($recentSearches as $search) {
                    echo '<tr>';
                    echo '<td>' . htmlspecialchars($search['city']) . '</td>';
                    echo '<td>' . htmlspecialchars($search['temperature']) . '</td>';
                    echo '<td>' . htmlspecialchars($search['search_date']) . '</td>';
                    echo '</tr>';
                }
            } else {
                echo '<tr><td colspan="3">Aucune recherche trouvée.</td></tr>';
            }
            ?>
        </tbody>
    </table>
</body>
</html>
