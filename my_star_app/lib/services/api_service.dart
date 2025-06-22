import 'dart:convert';
import 'dart:io';
import 'dart:math' as Math;
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = 'http://10.0.2.2:4000';
  
  static Future<Map<String, dynamic>> identifyStars(File imageFile, {int threshold = 300}) async {
    try {
      print('[DEBUG] Starting identifyStars request...');
      print('[DEBUG] Image file path: ${imageFile.path}');
      print('[DEBUG] Image file size: ${await imageFile.length()} bytes');
      print('[DEBUG] Threshold: $threshold');
      
      var request = http.MultipartRequest('POST', Uri.parse('$baseUrl/identify_fast'));
      print('[DEBUG] Request created for: $baseUrl/identify_fast');
      
      request.fields['threshold'] = threshold.toString();
      request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));
      print('[DEBUG] Request prepared with ${request.files.length} files and ${request.fields.length} fields');
      
      print('[DEBUG] Sending request...');
      var response = await request.send().timeout(
        const Duration(seconds: 300),  // Increased to 5 minutes
        onTimeout: () {
          print('[DEBUG] Request timed out after 5 minutes');
          throw Exception('Request timeout - server took too long to respond');
        },
      );
      
      print('[DEBUG] Response received with status: ${response.statusCode}');
      print('[DEBUG] Response headers: ${response.headers}');
      
      if (response.statusCode == 200) {
        print('[DEBUG] Reading response body...');
        String responseBody = await response.stream.bytesToString();
        print('[DEBUG] Response body length: ${responseBody.length} characters');
        print('[DEBUG] Response preview: ${responseBody.substring(0, Math.min(200, responseBody.length))}...');
        
        try {
          print('[DEBUG] Parsing JSON response...');
          final decodedResponse = json.decode(responseBody);
          print('[DEBUG] JSON parsed successfully');
          
          // Validate response structure for URL-based image response
          if (decodedResponse['image_url'] == null || decodedResponse['info'] == null) {
            print('[DEBUG] ERROR: Invalid response format - missing image_url or info');
            print('[DEBUG] Response keys: ${decodedResponse.keys.toList()}');
            throw Exception('Invalid response format');
          }
          
          print('[DEBUG] Response validation passed');
          print('[DEBUG] Image URL: ${decodedResponse['image_url']}');
          print('[DEBUG] Info keys: ${decodedResponse['info']?.keys?.toList() ?? []}');
          
          return decodedResponse;
        } catch (e) {
          print('[DEBUG] JSON decode error: $e');
          print('[DEBUG] Response body that failed to parse: ${responseBody.substring(0, Math.min(500, responseBody.length))}');
          throw Exception('Failed to parse server response: $e');
        }
      } else {
        print('[DEBUG] Server returned error status: ${response.statusCode}');
        final errorBody = await response.stream.bytesToString();
        print('[DEBUG] Error response body: $errorBody');
        throw Exception('Server error: ${response.statusCode}');
      }
    } catch (e) {
      print('[DEBUG] API Exception caught: $e');
      print('[DEBUG] Exception type: ${e.runtimeType}');
      throw Exception('Error: $e');
    }
  }
}