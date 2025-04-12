import React, { useState, useEffect } from "react";
import {
  AlertCircle,
  AlertTriangle,
  Check,
  User,
  Bot,
  HelpCircle,
  AlertOctagon,
} from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";

const AIDetectionVisualization = ({ aiDetection }) => {
  const [animationComplete, setAnimationComplete] = useState(false);

  // First, extract the score for fallback values
  const score =
    aiDetection?.ai_score !== undefined
      ? aiDetection.ai_score
      : aiDetection?.ai_likelihood !== undefined
      ? aiDetection.ai_likelihood
      : aiDetection?.ai_detection?.ai_score !== undefined
      ? aiDetection.ai_detection.ai_score
      : 0.5;

  // Determine fallback values based on score
  let fallbackEmoji = "❓";
  let fallbackIsAi = null;
  let fallbackMessage =
    "Unable to determine if content was written by AI or human";

  if (score >= 0.7) {
    fallbackEmoji = "🤖";
    fallbackIsAi = true;
    fallbackMessage = "This content was likely written by AI";
  } else if (score <= 0.3) {
    fallbackEmoji = "👤";
    fallbackIsAi = false;
    fallbackMessage = "This content was likely written by a human";
  }

  // Normalize the data structure regardless of how it comes in
  const normalizedData = {
    score: score,
    verdict:
      aiDetection?.ai_verdict ||
      aiDetection?.ai_detection?.ai_verdict ||
      "Unknown",
    emoji:
      aiDetection?.emoji || aiDetection?.ai_detection?.emoji || fallbackEmoji,
    displayMessage:
      aiDetection?.display_message ||
      aiDetection?.ai_detection?.display_message ||
      fallbackMessage,
    isAi:
      aiDetection?.is_ai !== undefined
        ? aiDetection.is_ai
        : aiDetection?.ai_detection?.is_ai !== undefined
        ? aiDetection.ai_detection.is_ai
        : fallbackIsAi,
    confidencePercentage:
      aiDetection?.confidence_percentage ||
      aiDetection?.ai_detection?.confidence_percentage ||
      Math.round(score * 100),
  };

  // This animation effect runs once when component mounts
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimationComplete(true);
    }, 1200);

    return () => clearTimeout(timer);
  }, []);

  const confidenceScore = normalizedData.confidencePercentage;

  // Determine message and styling based on AI detection result
  const getStyles = () => {
    if (normalizedData.isAi === true) {
      return {
        bgColor: "bg-red-50",
        borderColor: "border-red-200",
        textColor: "text-red-700",
        icon: <Bot className="text-red-500" size={24} />,
        title: "AI-Generated Content",
        primaryColor: "#ef4444", // red
      };
    } else if (normalizedData.isAi === false) {
      return {
        bgColor: "bg-green-50",
        borderColor: "border-green-200",
        textColor: "text-green-700",
        icon: <User className="text-green-500" size={24} />,
        title: "Human-Written Content",
        primaryColor: "#10b981", // green
      };
    } else {
      return {
        bgColor: "bg-gray-50",
        borderColor: "border-gray-200",
        textColor: "text-gray-700",
        icon: <HelpCircle className="text-gray-500" size={24} />,
        title: "Uncertain Origin",
        primaryColor: "#6b7280", // gray
      };
    }
  };

  const styles = getStyles();

  // Data for gauge chart
  const gaugeData = [
    { name: "AI Score", value: confidenceScore },
    { name: "Remaining", value: 100 - confidenceScore },
  ];

  // Only show if we have AI detection data
  if (!aiDetection) return null;

  return (
    <div
      className={`mb-8 rounded-lg overflow-hidden shadow-sm ${styles.borderColor} border`}
    >
      <div
        className={`p-4 ${styles.bgColor} border-b ${styles.borderColor} flex items-center justify-between`}
      >
        <div className="flex items-center">
          {styles.icon}
          <h3 className={`ml-2 font-bold ${styles.textColor}`}>
            {styles.title}
          </h3>
        </div>
        <div
          className={`px-2 py-1 rounded-full text-sm flex items-center ${
            confidenceScore > 75
              ? "bg-opacity-20 bg-red-500 text-red-800"
              : confidenceScore > 40
              ? "bg-opacity-20 bg-yellow-500 text-yellow-800"
              : "bg-opacity-20 bg-green-500 text-green-800"
          }`}
        >
          {confidenceScore}%{" "}
          {confidenceScore > 75
            ? "High"
            : confidenceScore > 40
            ? "Medium"
            : "Low"}{" "}
          Confidence
        </div>
      </div>

      <div className="p-6 bg-white">
        <div className="flex flex-col md:flex-row items-center">
          {/* Left side - big emoji */}
          <div className="text-6xl mb-4 md:mb-0 md:mr-6 flex-shrink-0 w-24 h-24 rounded-full flex items-center justify-center">
            <div
              className={`${
                animationComplete ? "scale-100" : "scale-0"
              } transition-transform duration-500`}
            >
              {normalizedData.emoji}
            </div>
          </div>

          {/* Right side - gauge and text */}
          <div className="flex-1">
            <h4 className="text-lg font-medium mb-2">
              {normalizedData.displayMessage}
            </h4>
            <p className="text-gray-600 mb-4">{normalizedData.verdict}</p>

            <div className="h-32 mb-4">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={gaugeData}
                    startAngle={180}
                    endAngle={0}
                    innerRadius="60%"
                    outerRadius="80%"
                    paddingAngle={0}
                    dataKey="value"
                    isAnimationActive={true}
                    animationDuration={1000}
                  >
                    <Cell fill={styles.primaryColor} />
                    <Cell fill="#e5e7eb" /> {/* Light gray for remaining */}
                  </Pie>
                  <text
                    x="50%"
                    y="50%"
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className="font-bold"
                    fill={styles.primaryColor}
                  >
                    {confidenceScore}%
                  </text>
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="text-sm text-gray-500">
              {normalizedData.isAi === true ? (
                <div className="flex items-start">
                  <AlertOctagon
                    size={16}
                    className="mr-2 mt-0.5 flex-shrink-0 text-red-500"
                  />
                  <span>
                    This content shows strong indicators of AI generation based
                    on linguistic patterns, structural consistency, and content
                    analysis.
                  </span>
                </div>
              ) : normalizedData.isAi === false ? (
                <div className="flex items-start">
                  <Check
                    size={16}
                    className="mr-2 mt-0.5 flex-shrink-0 text-green-500"
                  />
                  <span>
                    This content displays natural language patterns consistent
                    with human writing, including varied paragraph lengths,
                    personal pronouns, and natural language flow.
                  </span>
                </div>
              ) : (
                <div className="flex items-start">
                  <AlertTriangle
                    size={16}
                    className="mr-2 mt-0.5 flex-shrink-0 text-yellow-500"
                  />
                  <span>
                    The analysis shows mixed indicators, making it difficult to
                    confidently determine whether this content was created by AI
                    or human.
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIDetectionVisualization;
