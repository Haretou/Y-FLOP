<?php
require_once 'api/api.php';
require_once 'crud/read.php';
require_once 'crud/create.php';

$weather = null;

if (!empty($_POST['city'])) {
    $city = $_POST['city'];
    $weather = fetchWeatherData('current.json', ['q' => $city]);

    if (isset($weather['current'])) {
        saveSearch($city, $weather['current']['temp_c'], $weather['current']['condition']['text']);
    }
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Site Météo</title>
    <link rel="stylesheet" href="assets/css/styles.css">
</head>
<body>
    <h1>Recherchez la météo</h1>
    <form method="POST">
        <input type="text" name="city" placeholder="Entrez une ville" required>
        <button type="submit">Rechercher</button>
    </form>

    <?php if ($weather): ?>
        <h2>Météo pour <?= htmlspecialchars($city) ?></h2>
        <p>Température : <?= $weather['current']['temp_c'] ?> °C</p>
        <p>Condition : <?= $weather['current']['condition']['text'] ?></p>
    <?php endif; ?>

    <h2>Dernières recherches</h2>
    <ul>
        <?php foreach (getRecentSearches() as $search): ?>
            <li><?= htmlspecialchars($search['city']) ?> : <?= $search['temperature'] ?> °C (<?= $search['description'] ?>)</li>
        <?php endforeach; ?>
    </ul>
</body>
</html>
