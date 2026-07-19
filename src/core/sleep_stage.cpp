#include "eeg_to_hypnogram/sleep_stage.h"

#include <cctype>
#include <stdexcept>
#include <string>

namespace eeg_to_hypnogram
{
    namespace
    {

        std::string NormalizeStageText(std::string_view text)
        {
            std::string normalized;
            normalized.reserve(text.size());

            bool lastWasSpace = true;

            for (const unsigned char character : text)
            {
                const bool isSpace =
                    std::isspace(character) != 0;

                if (isSpace)
                {
                    if (!lastWasSpace)
                    {
                        normalized.push_back(' ');
                        lastWasSpace = true;
                    }

                    continue;
                }

                normalized.push_back(
                    static_cast<char>(
                        std::tolower(character)));

                lastWasSpace = false;
            }

            if (!normalized.empty() &&
                normalized.back() == ' ')
            {
                normalized.pop_back();
            }

            return normalized;
        }

        bool StartsWith(
            const std::string &text,
            const std::string &prefix)
        {
            return text.size() >= prefix.size() &&
                   text.compare(
                       0,
                       prefix.size(),
                       prefix) == 0;
        }

        bool TryParseStageToken(
            const std::string &token,
            SleepStage &stage)
        {
            if (token == "w" ||
                token == "wake")
            {
                stage = SleepStage::Wake;
                return true;
            }

            if (token == "1" ||
                token == "n1")
            {
                stage = SleepStage::N1;
                return true;
            }

            if (token == "2" ||
                token == "n2")
            {
                stage = SleepStage::N2;
                return true;
            }

            if (token == "3" ||
                token == "4" ||
                token == "n3" ||
                token == "n4")
            {
                stage = SleepStage::N3;
                return true;
            }

            if (token == "r" ||
                token == "rem")
            {
                stage = SleepStage::Rem;
                return true;
            }

            if (token == "?" ||
                token == "unknown")
            {
                stage = SleepStage::Unknown;
                return true;
            }

            if (token == "movement")
            {
                stage = SleepStage::Movement;
                return true;
            }

            return false;
        }

    } // namespace

    SleepStage ParseSleepStage(
        std::string_view text)
    {
        SleepStage stage = SleepStage::Unknown;

        if (TryParseSleepStage(text, stage))
        {
            return stage;
        }

        return SleepStage::Unknown;
    }

    bool TryParseSleepStage(
        std::string_view text,
        SleepStage &stage)
    {
        const std::string normalized =
            NormalizeStageText(text);

        if (normalized.empty())
        {
            return false;
        }

        if (normalized == "movement time")
        {
            stage = SleepStage::Movement;
            return true;
        }

        constexpr char kSleepStagePrefix[] = "sleep stage ";

        if (StartsWith(normalized, kSleepStagePrefix))
        {
            const std::string token =
                normalized.substr(
                    std::char_traits<char>::length(
                        kSleepStagePrefix));

            return TryParseStageToken(
                token,
                stage);
        }

        return TryParseStageToken(
            normalized,
            stage);
    }

    std::string SleepStageToString(
        SleepStage stage)
    {
        switch (stage)
        {
        case SleepStage::Wake:
            return "W";

        case SleepStage::N1:
            return "N1";

        case SleepStage::N2:
            return "N2";

        case SleepStage::N3:
            return "N3";

        case SleepStage::Rem:
            return "REM";

        case SleepStage::Unknown:
            return "UNKNOWN";

        case SleepStage::Movement:
            return "MOVEMENT";
        }

        throw std::invalid_argument(
            "Unsupported sleep stage enum value.");
    }

    bool IsTrainableSleepStage(
        SleepStage stage)
    {
        switch (stage)
        {
        case SleepStage::Wake:
        case SleepStage::N1:
        case SleepStage::N2:
        case SleepStage::N3:
        case SleepStage::Rem:
            return true;

        case SleepStage::Unknown:
        case SleepStage::Movement:
            return false;
        }

        return false;
    }

    int SleepStageToClassLabel(
        SleepStage stage)
    {
        switch (stage)
        {
        case SleepStage::Wake:
            return 0;

        case SleepStage::N1:
            return 1;

        case SleepStage::N2:
            return 2;

        case SleepStage::N3:
            return 3;

        case SleepStage::Rem:
            return 4;

        case SleepStage::Unknown:
            throw std::invalid_argument(
                "UNKNOWN sleep stage is not trainable.");

        case SleepStage::Movement:
            throw std::invalid_argument(
                "MOVEMENT sleep stage is not trainable.");
        }

        throw std::invalid_argument(
            "Unsupported sleep stage enum value.");
    }

    SleepStage ClassLabelToSleepStage(
        int label)
    {
        switch (label)
        {
        case 0:
            return SleepStage::Wake;

        case 1:
            return SleepStage::N1;

        case 2:
            return SleepStage::N2;

        case 3:
            return SleepStage::N3;

        case 4:
            return SleepStage::Rem;

        default:
            throw std::invalid_argument(
                "Sleep stage class label must be in [0, 4].");
        }
    }

} // namespace eeg_to_hypnogram
