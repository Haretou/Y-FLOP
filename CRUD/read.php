<?php
require_once __DIR__ . '/../config.php';  // Inclure la connexion à la base de données

// Obtenir la connexion PDO
$pdo = getDatabaseConnection();

function getRecentSearches($pdo, $limit = 10) {
    if (!$pdo) {
        echo "Erreur de connexion à la base de données!";
        return [];
    }

    try {
        // Préparer et exécuter la requête pour récupérer les recherches récentes
        $stmt = $pdo->prepare("SELECT city, temperature, description FROM searches ORDER BY created_at DESC LIMIT ?");
        $stmt->execute([$limit]);
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    } catch (PDOException $e) {
        echo "Erreur SQL: " . $e->getMessage();
        return [];
    }
}

if ($pdo) {
    $query = "
        SELECT 
            Place.Name AS Place_Name, 
            Place.Latitude, 
            Place.Longitude, 
            Weather.State, 
            Weather.Town, 
            Weather.temperature, 
            Weather.precipitation, 
            Weather.wind, 
            Weather.humidity, 
            Time.day, 
            Time.hour
        FROM Place
        JOIN Time ON Place.id = Time.place_id
        JOIN Weather ON Time.weather_id = Weather.id
    ";

    try {
        $statement = $pdo->query($query);

        $results = $statement->fetchAll(PDO::FETCH_ASSOC);

        // Commenter ou supprimer la partie qui génère le tableau
        /* if ($results) {
            echo "<h2>Informations météorologiques</h2>";
            echo "<table border='1'>
                    <tr>
                        <th>Nom du Lieu</th>
                        <th>Latitude</th>
                        <th>Longitude</th>
                        <th>État</th>
                        <th>Ville</th>
                        <th>Température</th>
                        <th>Précipitations</th>
                        <th>Vent</th>
                        <th>Humidité</th>
                        <th>Jour</th>
                        <th>Heure</th>
                    </tr>";

            foreach ($results as $row) {
                echo "<tr>
                        <td>" . htmlspecialchars($row['Place_Name']) . "</td>
                        <td>" . htmlspecialchars($row['Latitude']) . "</td>
                        <td>" . htmlspecialchars($row['Longitude']) . "</td>
                        <td>" . htmlspecialchars($row['State']) . "</td>
                        <td>" . htmlspecialchars($row['Town']) . "</td>
                        <td>" . htmlspecialchars($row['temperature']) . "°C</td>
                        <td>" . htmlspecialchars($row['precipitation']) . " mm</td>
                        <td>" . htmlspecialchars($row['wind']) . " km/h</td>
                        <td>" . htmlspecialchars($row['humidity']) . "%</td>
                        <td>" . htmlspecialchars($row['day']) . "</td>
                        <td>" . htmlspecialchars($row['hour']) . "</td>
                    </tr>";
            }
            echo "</table>";
        } else {
            echo "Aucune donnée disponible.";
        } */

    } catch (PDOException $e) {
        echo "Erreur SQL: " . $e->getMessage();
    }
} else {
    echo "Erreur de connexion à la base de données!";
}
?>
