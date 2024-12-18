<?php
// Inclure le fichier de configuration pour la connexion à la base de données
require_once __DIR__ . '/../config.php';

try {
    // Préparer la requête SQL avec des jointures pour récupérer les informations sur le lieu, la température et l'heure
    $query = "
        SELECT p.city, w.temperature, t.time 
        FROM weather w
        JOIN place p ON w.place_id = p.id
        JOIN time t ON w.time_id = t.id
        ORDER BY t.time DESC
        LIMIT 5
    ";

    // Exécuter la requête
    $statement = $pdo->query($query);

    // Récupérer les résultats sous forme de tableau associatif
    $results = $statement->fetchAll(PDO::FETCH_ASSOC);

    // Vérifier si des résultats ont été trouvés
    if ($results) {
        echo "<h2>Dernières recherches</h2>";
        echo "<table border='1'>
                <tr>
                    <th>Ville</th>
                    <th>Température</th>
                    <th>Heure</th>
                </tr>";

        // Afficher les résultats sous forme de tableau
        foreach ($results as $row) {
            echo "<tr>
                    <td>" . htmlspecialchars($row['city']) . "</td>
                    <td>" . htmlspecialchars($row['temperature']) . "°C</td>
                    <td>" . htmlspecialchars($row['time']) . "</td>
                  </tr>";
        }

        echo "</table>";
    } else {
        echo "Aucune donnée trouvée.";
    }
} catch (PDOException $e) {
    echo "Erreur : " . $e->getMessage();
}
?>
