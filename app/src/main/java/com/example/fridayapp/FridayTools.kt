package com.example.fridayapp

import android.content.Context
import android.content.Intent
import android.hardware.camera2.CameraManager
import android.net.Uri
import android.os.BatteryManager
import android.provider.AlarmClock
import android.provider.MediaStore
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.TimeUnit

object FridayTools {
    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()

    // TODO: You should fetch this from a secure location.
    private const val api_key = "AIzaSyDL0Un9qw-EAfYYrix2l0XH-rtl-O6Y15A"
    private const val cx_id = "207ac63419d184fc2"


    fun execute(toolName: String, args: JSONObject, context: Context): String{
        return try{
            when (toolName) {
                "get_time" -> getCurrentTime()
                "google_search" -> googleSearch(args.optString("query"))
                "weather" -> getWeather(args.optString("location"))
                "wikipedia" -> searchWikipedia(args.optString("query"))
                "parse_url" -> parseUrl(args.optString("url"))
                "device_hardware" -> handleDeviceHardware(args, context)
                "set_timer" -> setTimer(context, args.getInt("duration"), args.optString("message", "Timer"))
                "set_alarm" -> setAlarm(context, args.optString("time"), args.optString("label"))
                else -> "Invalid tool name"
            }
        } catch (e: Exception){
            "Error executing tool: ${e.javaClass.simpleName} - ${e.message}"
        }
    }

    private fun getCurrentTime(): String {
        val sdf = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
        return "Current Date and Time: ${sdf.format(Date())}"
    }

    private fun googleSearch(query: String): String {
        if (query.isEmpty()) {
            return "Please provide a search query."
        }

        val url = "https://www.googleapis.com/customsearch/v1?key=$api_key&cx=$cx_id&q=$query"

        val request = Request.Builder().url(url).build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                return "Error: ${response.code}"
            }

            val jsonResponse = JSONObject(response.body?.string() ?: "")
            val items = jsonResponse.optJSONArray("items")
            if (items == null || items.length() == 0) {
                return "No search results found."
            }

            val sb = StringBuilder()
            sb.append("Search results for: $query")

            for (i in 0 until minOf(items.length(), 3)) {
                val item = items.getJSONObject(i)
                val title = item.getString("title")
                val snippet = item.getString("snippet")
                sb.append("${i + 1}. $title: $snippet")

            }
            return sb.toString()

        }
    }

    private fun getWeather(location: String): String {
        if (location.isEmpty()) {
            return "Please provide a location for the weather forecast."
        }
        try {
            // 1. Geocode location to get lat/lon
            val geocodeUrl = "https://geocode.maps.co/search?q=${Uri.encode(location)}"
            val geocodeRequest = Request.Builder().url(geocodeUrl).build()
            var lat = 0.0
            var lon = 0.0

            client.newCall(geocodeRequest).execute().use { response ->
                if (!response.isSuccessful) return "Could not find location: $location"
                val jsonResponse = JSONArray(response.body?.string() ?: "[]")
                if (jsonResponse.length() == 0) return "Could not find location: $location"
                val firstResult = jsonResponse.getJSONObject(0)
                lat = firstResult.getDouble("lat")
                lon = firstResult.getDouble("lon")
            }

            // 2. Get weather from Open-Meteo
            val weatherUrl = "https://api.open-meteo.com/v1/forecast?latitude=$lat&longitude=$lon&current=temperature_2m,wind_speed_10m,weather_code&daily=weather_code,temperature_2m_max,temperature_2m_min&timezone=auto"
            val weatherRequest = Request.Builder().url(weatherUrl).build()

            client.newCall(weatherRequest).execute().use { response ->
                if (!response.isSuccessful) return "Error fetching weather data."
                val weatherJson = JSONObject(response.body?.string() ?: "{}")

                val current = weatherJson.getJSONObject("current")
                val daily = weatherJson.getJSONObject("daily")

                val temp = current.getDouble("temperature_2m")
                val wind = current.getDouble("wind_speed_10m")
                val weatherCode = current.getInt("weather_code")
                val maxTemp = daily.getJSONArray("temperature_2m_max").getDouble(0)
                val minTemp = daily.getJSONArray("temperature_2m_min").getDouble(0)
                val units = weatherJson.getJSONObject("current_units")
                val tempUnit = units.getString("temperature_2m")
                val windUnit = units.getString("wind_speed_10m")


                return "Weather for $location: Currently ${getWeatherDescription(weatherCode)} with a temperature of $temp$tempUnit. " +
                       "High of $maxTemp$tempUnit and a low of $minTemp$tempUnit. Wind speed is $wind$windUnit."
            }
        } catch (e: Exception) {
            return "Error getting weather: ${e.message}"
        }
    }

    private fun getWeatherDescription(code: Int): String {
        return when (code) {
            0 -> "Clear sky"
            1, 2, 3 -> "Mainly clear, partly cloudy, and overcast"
            45, 48 -> "Fog and depositing rime fog"
            51, 53, 55 -> "Drizzle: Light, moderate, and dense intensity"
            56, 57 -> "Freezing Drizzle: Light and dense intensity"
            61, 63, 65 -> "Rain: Slight, moderate and heavy intensity"
            66, 67 -> "Freezing Rain: Light and heavy intensity"
            71, 73, 75 -> "Snow fall: Slight, moderate, and heavy intensity"
            77 -> "Snow grains"
            80, 81, 82 -> "Rain showers: Slight, moderate, and violent"
            85, 86 -> "Snow showers slight and heavy"
            95 -> "Thunderstorm: Slight or moderate"
            96, 99 -> "Thunderstorm with slight and heavy hail"
            else -> "Unknown weather"
        }
    }


    private fun searchWikipedia(query: String): String {
        if (query.isEmpty()) {
            return "Please provide a query for Wikipedia."
        }

        try {
            val encodedQuery = Uri.encode(query)
            val url = "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&redirects=1&titles=$encodedQuery"

            val request = Request.Builder().url(url).build()

            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    return "Error parsing Wikipedia: ${response.code}"
                }

                val responseBody = response.body?.string() ?: return "Empty response from Wikipedia."
                val json = JSONObject(responseBody)
                val queryObj = json.optJSONObject("query") ?: return "No query data found."
                val pages = queryObj.optJSONObject("pages") ?: return "No pages found."

                val pageId = pages.keys().next()

                if (pageId == "-1") {
                    return "No Wikipedia article found for '$query'."
                }

                val page = pages.getJSONObject(pageId)
                val title = page.optString("title", "Unknown Title")
                val extract = page.optString("extract", "No content available.")

                if (extract.isEmpty()) {
                    return "Found page '$title', but it contains no summary text."
                }

                return "$title: $extract"
            }

        } catch (e: Exception) {
            return "Error searching Wikipedia: ${e.message}"
        }
    }

    private fun parseUrl(url: String): String {
        if (url.isEmpty()) {
            return "Please provide a url."
        }
        val request = Request.Builder().url(url).build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                return "Error: ${response.code}"
            }

            val sb = StringBuilder()
            sb.append("Parsing URL: $url")
            sb.append("Status Code: ${response.code}")
            sb.append("Content Type: ${response.header("Content-Type")}")

            val contentLength = response.header("Content-Length")

            if (contentLength != null) {
                sb.append("Content Length: $contentLength")
            } else {
                sb.append("Content Length: Unknown")
            }

            val responseBody = response.body?.string()
            if (responseBody != null) {
                sb.append("Response Body (first 500 chars):${responseBody.take(500)}")
            } else {
                sb.append("Response Body: Empty")
            }
            return sb.toString()
        }
    }

    private fun handleDeviceHardware(args: JSONObject, context: Context): String {
        val action = args.optString("action")
        if (action.isEmpty()) {
            return "Invalid hardware action: action not specified."
        }
        return when (action) {
            "open_camera" -> {
                val intent = Intent(MediaStore.INTENT_ACTION_STILL_IMAGE_CAMERA).apply {
                    addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                if (intent.resolveActivity(context.packageManager) != null) {
                    context.startActivity(intent)
                    "Opening camera."
                } else {
                    "Could not open camera. No app found to handle it."
                }
            }
            "toggle_flashlight" -> {
                val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
                val cameraId = cameraManager.cameraIdList[0]
                val enable = args.getBoolean("enable")
                try {
                    cameraManager.setTorchMode(cameraId, enable)
                    if (enable) "Flashlight turned on." else "Flashlight turned off."
                } catch(e: Exception) {
                    "Error toggling flashlight: ${e.message}"
                }
            }
            "get_battery_level" -> {
                 val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
                 val batLevel = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
                "Current battery level is $batLevel%"
            }
            else -> "Invalid hardware action."
        }
    }

    private fun setTimer(context: Context, seconds: Int, message: String): String {
        val intent = Intent(AlarmClock.ACTION_SET_TIMER).apply {
            putExtra(AlarmClock.EXTRA_MESSAGE, message)
            putExtra(AlarmClock.EXTRA_LENGTH, seconds)
            putExtra(AlarmClock.EXTRA_SKIP_UI, false) // Show UI for confirmation
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        }
        if (intent.resolveActivity(context.packageManager) != null) {
             context.startActivity(intent)
             return "Timer for $seconds seconds with message '$message' has been set."
        } else {
            return "Could not set timer. No app found to handle it."
        }
    }

    private fun setAlarm(context: Context, time: String, label: String): String {
        if (time.isEmpty()) {
            return "Error setting alarm: time not specified. Please use format HH:mm."
        }
        try {
            val timeParts = time.split(":")
            val hour = timeParts[0].toInt()
            val minute = timeParts[1].toInt()

            val intent = Intent(AlarmClock.ACTION_SET_ALARM).apply {
                putExtra(AlarmClock.EXTRA_MESSAGE, label)
                putExtra(AlarmClock.EXTRA_HOUR, hour)
                putExtra(AlarmClock.EXTRA_MINUTES, minute)
                putExtra(AlarmClock.EXTRA_SKIP_UI, false) // Show UI for confirmation
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            if (intent.resolveActivity(context.packageManager) != null) {
                context.startActivity(intent)
                return "Alarm set for $time with label '$label'."
            } else {
                return "Could not set alarm. No app found to handle it."
            }
        } catch (e: Exception) {
            return "Error setting alarm: ${e.message}. Please use format HH:mm."
        }
    }
}
