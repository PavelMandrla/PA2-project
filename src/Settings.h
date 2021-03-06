#ifndef PA2PROJECT_SETTINGS_H
#define PA2PROJECT_SETTINGS_H

#include <string>

using namespace std;

class Settings {
private:
    string getAtribVal(string &str, const string& atribName);
public:
    int leaders {0};
    int followers {0};
    string heightMap;
    int heightmapGridX {0};
    int heightmapGridY {0};
    int viewportWidth {1024};
    int viewportHeight {1024};
    float leaderRadius {0.0f};
    float speedFactor {0.0f};
    string outputFile;

    Settings(const string& path);
};


#endif //PA2PROJECT_SETTINGS_H
