const int numLanes = 3;

// 引脚定义
// const int laneInputPins[numLanes] = {3, 4, 5};  
// const int laneOutputPin1[numLanes] = {6, 8, 10};
// const int laneOutputPin2[numLanes] = {7, 9, 11};
const int laneInputPins[numLanes] = {5, 4, 3};  
const int laneOutputPin1[numLanes] = {10, 8, 6};
const int laneOutputPin2[numLanes] = {11, 9, 7};
const int LED = 13;

// 参数设置
const unsigned long debounceDelay = 15;
const unsigned long ledBlinkTime = 150;

// --- 队列逻辑修改部分 ---
const int QUEUE_SIZE = 15; // 每个通道最多排队15个apple
int laneQueues[numLanes][QUEUE_SIZE]; 
int head[numLanes] = {0, 0, 0}; // 指向当前要处理的任务
int tail[numLanes] = {0, 0, 0}; // 指向下一个存放新指令的位置

// 状态存储
bool lastStableState[numLanes];
bool currentState[numLanes];
unsigned long lastDebounceTime[numLanes];
unsigned long ledOnTime = 0;
bool ledState = false;

void setup() {
  Serial.begin(115200);
  pinMode(LED, OUTPUT);
  digitalWrite(LED, LOW);

  for (int i = 0; i < numLanes; i++) {
    pinMode(laneInputPins[i], INPUT_PULLUP);
    pinMode(laneOutputPin1[i], OUTPUT);
    pinMode(laneOutputPin2[i], OUTPUT);
        
    digitalWrite(laneOutputPin1[i], LOW);
    digitalWrite(laneOutputPin2[i], LOW);
    
    lastStableState[i] = HIGH;
    currentState[i] = HIGH;
    lastDebounceTime[i] = 0;
    
    // 初始化队列为0
    for(int j=0; j<QUEUE_SIZE; j++) laneQueues[i][j] = 0;
  }
}

void loop() {
  unsigned long currentMillis = millis();

  // ================= 1. 串口读取：进队 (Tail++) =================
  while (Serial.available() > 0) {
    char c = Serial.read();
    static String inputBuffer = "";
    
    if (c == '\n') {
      inputBuffer.trim();
      if (inputBuffer.length() == numLanes) {
        for (int i = 0; i < numLanes; i++) {
          int actionInput = inputBuffer.charAt(i) - '0';
          // 只有当任务不为 '0' 时才存入队列
          if (actionInput > 0 && actionInput <= 3) {
            int nextTail = (tail[i] + 1) % QUEUE_SIZE;
            // 检查队列是否已满，没满则存入
            if (nextTail != head[i]) {
              laneQueues[i][tail[i]] = actionInput;
              tail[i] = nextTail;
            }
          }
        }
      }
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }

  // ================= 2. 传感器检测与出队 (Head++) =================
  bool triggered = false;
  for (int i = 0; i < numLanes; i++) {
    // 检查是否有任务在排队
    if (head[i] != tail[i]) {
      // 感应器检测触发
      if (obDetect(i)) {
        int currentAction = laneQueues[i][head[i]];
        doAction(i, currentAction);
        
        // 任务完成，出队
        head[i] = (head[i] + 1) % QUEUE_SIZE;
        triggered = true;
      }
    }
  }

  // ================= 3. 反馈与LED =================
  if (triggered) {
    Serial.println("OK"); // 告诉Python执行了一个动作
    digitalWrite(LED, HIGH);
    ledState = true;
    ledOnTime = currentMillis;
  }

  if (ledState && (currentMillis - ledOnTime >= ledBlinkTime)) {
    digitalWrite(LED, LOW);
    ledState = false;
  }
}

// ================= 输入检测 =================
int obDetect(int lane) {
  int reading = digitalRead(laneInputPins[lane]);
  if (reading != lastStableState[lane]) {
    lastDebounceTime[lane] = millis();
  }
  if ((millis() - lastDebounceTime[lane]) > debounceDelay) {
    if (reading != currentState[lane]) {
      currentState[lane] = reading;
      if (currentState[lane] == LOW) {
        lastStableState[lane] = currentState[lane];
        return 1;
      }
    }
  }
  lastStableState[lane] = reading;
  return 0;
}
// ================= 动作执行 =================
void doAction(int lane, int actionType) {
  switch (actionType) {
    case 1:
      digitalWrite(laneOutputPin1[lane], LOW);
      digitalWrite(laneOutputPin2[lane], LOW);
      break;

    case 2:
      digitalWrite(laneOutputPin1[lane], HIGH);
      digitalWrite(laneOutputPin2[lane], HIGH);
      break;

    case 3:
      digitalWrite(laneOutputPin1[lane], LOW);
      digitalWrite(laneOutputPin2[lane], HIGH);
      break;

    default:
      digitalWrite(laneOutputPin1[lane], LOW);
      digitalWrite(laneOutputPin2[lane], LOW);
      break;
  }
}
