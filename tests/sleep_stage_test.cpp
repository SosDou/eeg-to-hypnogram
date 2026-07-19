#include "eeg_to_hypnogram/sleep_stage.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    int g_caseCount = 0;

    void Require(
        bool condition,
        const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(
                "Test failed: " + message);
        }
    }

    template <typename ExceptionType, typename Function>
    void RequireThrows(
        Function &&function,
        const std::string &message)
    {
        try
        {
            function();
        }
        catch (const ExceptionType &)
        {
            return;
        }
        catch (const std::exception &exception)
        {
            throw std::runtime_error(
                "Test failed: " +
                message +
                "; unexpected exception: " +
                exception.what());
        }

        throw std::runtime_error(
            "Test failed: " +
            message +
            "; no exception was thrown.");
    }

    void Pass(
        const std::string &name)
    {
        ++g_caseCount;
        std::cout
            << "[PASS] "
            << name
            << '\n';
    }

    void RequireParsed(
        const std::string &text,
        eeg_to_hypnogram::SleepStage expected,
        const std::string &message)
    {
        eeg_to_hypnogram::SleepStage stage =
            eeg_to_hypnogram::SleepStage::Unknown;

        Require(
            eeg_to_hypnogram::TryParseSleepStage(
                text,
                stage),
            message + " must parse.");

        Require(
            stage == expected,
            message + " parsed to the wrong stage.");
    }

    void TestCanonicalStageParsing()
    {
        using eeg_to_hypnogram::SleepStage;

        RequireParsed("W", SleepStage::Wake, "W");
        RequireParsed("N1", SleepStage::N1, "N1");
        RequireParsed("N2", SleepStage::N2, "N2");
        RequireParsed("N3", SleepStage::N3, "N3");
        RequireParsed("N4", SleepStage::N3, "N4");
        RequireParsed("REM", SleepStage::Rem, "REM");
        RequireParsed("R", SleepStage::Rem, "R");
        RequireParsed("UNKNOWN", SleepStage::Unknown, "UNKNOWN");
        RequireParsed("MOVEMENT", SleepStage::Movement, "MOVEMENT");

        RequireParsed("Wake", SleepStage::Wake, "Wake");
        RequireParsed("Rem", SleepStage::Rem, "Rem");
        RequireParsed("Unknown", SleepStage::Unknown, "Unknown");
        RequireParsed("Movement", SleepStage::Movement, "Movement");

        Pass("canonical stage parsing");
    }

    void TestSleepEdfAnnotationParsing()
    {
        using eeg_to_hypnogram::SleepStage;

        RequireParsed("Sleep stage W", SleepStage::Wake, "Sleep stage W");
        RequireParsed("Sleep stage 1", SleepStage::N1, "Sleep stage 1");
        RequireParsed("Sleep stage 2", SleepStage::N2, "Sleep stage 2");
        RequireParsed("Sleep stage 3", SleepStage::N3, "Sleep stage 3");
        RequireParsed("Sleep stage 4", SleepStage::N3, "Sleep stage 4");
        RequireParsed("Sleep stage R", SleepStage::Rem, "Sleep stage R");
        RequireParsed("Sleep stage ?", SleepStage::Unknown, "Sleep stage ?");
        RequireParsed("Movement time", SleepStage::Movement, "Movement time");

        Pass("Sleep-EDF annotation parsing");
    }

    void TestWhitespaceAndCaseNormalization()
    {
        using eeg_to_hypnogram::SleepStage;

        RequireParsed(
            "  sleep   STAGE   w  ",
            SleepStage::Wake,
            "spaced wake annotation");

        RequireParsed(
            "\tN2\n",
            SleepStage::N2,
            "trimmed N2");

        RequireParsed(
            "rEm",
            SleepStage::Rem,
            "mixed-case REM");

        RequireParsed(
            "Movement    Time",
            SleepStage::Movement,
            "spaced movement annotation");

        Pass("whitespace and case normalization");
    }

    void TestN4MergeRule()
    {
        using eeg_to_hypnogram::SleepStage;

        RequireParsed("N4", SleepStage::N3, "N4");
        RequireParsed("Sleep stage 4", SleepStage::N3, "Sleep stage 4");
        RequireParsed("Sleep stage N4", SleepStage::N3, "Sleep stage N4");

        Require(
            eeg_to_hypnogram::SleepStageToString(
                eeg_to_hypnogram::ParseSleepStage("N4")) == "N3",
            "N4 must normalize to N3.");

        Pass("N4 merge rule");
    }

    void TestUnknownAndMovementParsing()
    {
        using eeg_to_hypnogram::SleepStage;

        RequireParsed("Sleep stage ?", SleepStage::Unknown, "Sleep stage ?");
        RequireParsed("?", SleepStage::Unknown, "?");
        RequireParsed("unknown", SleepStage::Unknown, "unknown");
        RequireParsed("Movement time", SleepStage::Movement, "Movement time");
        RequireParsed("movement", SleepStage::Movement, "movement");

        Pass("unknown and movement parsing");
    }

    void TestNonStageAnnotationRejection()
    {
        eeg_to_hypnogram::SleepStage stage =
            eeg_to_hypnogram::SleepStage::Wake;

        Require(
            !eeg_to_hypnogram::TryParseSleepStage(
                "Lights off",
                stage),
            "ordinary EDF annotation must be rejected.");

        Require(
            stage == eeg_to_hypnogram::SleepStage::Wake,
            "failed TryParseSleepStage must not modify output.");

        Require(
            eeg_to_hypnogram::ParseSleepStage(
                "Lights off") ==
                eeg_to_hypnogram::SleepStage::Unknown,
            "ParseSleepStage must fall back to Unknown.");

        Pass("non-stage annotation rejection");
    }

    void TestCanonicalStringConversion()
    {
        using eeg_to_hypnogram::SleepStage;
        using eeg_to_hypnogram::SleepStageToString;

        Require(SleepStageToString(SleepStage::Wake) == "W", "Wake string changed.");
        Require(SleepStageToString(SleepStage::N1) == "N1", "N1 string changed.");
        Require(SleepStageToString(SleepStage::N2) == "N2", "N2 string changed.");
        Require(SleepStageToString(SleepStage::N3) == "N3", "N3 string changed.");
        Require(SleepStageToString(SleepStage::Rem) == "REM", "REM string changed.");
        Require(SleepStageToString(SleepStage::Unknown) == "UNKNOWN", "Unknown string changed.");
        Require(SleepStageToString(SleepStage::Movement) == "MOVEMENT", "Movement string changed.");

        Pass("canonical string conversion");
    }

    void TestTrainableStageRules()
    {
        using eeg_to_hypnogram::IsTrainableSleepStage;
        using eeg_to_hypnogram::SleepStage;

        Require(IsTrainableSleepStage(SleepStage::Wake), "Wake must be trainable.");
        Require(IsTrainableSleepStage(SleepStage::N1), "N1 must be trainable.");
        Require(IsTrainableSleepStage(SleepStage::N2), "N2 must be trainable.");
        Require(IsTrainableSleepStage(SleepStage::N3), "N3 must be trainable.");
        Require(IsTrainableSleepStage(SleepStage::Rem), "Rem must be trainable.");
        Require(!IsTrainableSleepStage(SleepStage::Unknown), "Unknown must not be trainable.");
        Require(!IsTrainableSleepStage(SleepStage::Movement), "Movement must not be trainable.");

        Pass("trainable stage rules");
    }

    void TestClassLabelConversion()
    {
        using eeg_to_hypnogram::ClassLabelToSleepStage;
        using eeg_to_hypnogram::SleepStage;
        using eeg_to_hypnogram::SleepStageToClassLabel;

        Require(SleepStageToClassLabel(SleepStage::Wake) == 0, "Wake label changed.");
        Require(SleepStageToClassLabel(SleepStage::N1) == 1, "N1 label changed.");
        Require(SleepStageToClassLabel(SleepStage::N2) == 2, "N2 label changed.");
        Require(SleepStageToClassLabel(SleepStage::N3) == 3, "N3 label changed.");
        Require(SleepStageToClassLabel(SleepStage::Rem) == 4, "Rem label changed.");

        Require(ClassLabelToSleepStage(0) == SleepStage::Wake, "0 stage changed.");
        Require(ClassLabelToSleepStage(1) == SleepStage::N1, "1 stage changed.");
        Require(ClassLabelToSleepStage(2) == SleepStage::N2, "2 stage changed.");
        Require(ClassLabelToSleepStage(3) == SleepStage::N3, "3 stage changed.");
        Require(ClassLabelToSleepStage(4) == SleepStage::Rem, "4 stage changed.");

        Pass("class label conversion");
    }

    void TestInvalidClassConversion()
    {
        using eeg_to_hypnogram::ClassLabelToSleepStage;
        using eeg_to_hypnogram::SleepStage;
        using eeg_to_hypnogram::SleepStageToClassLabel;

        RequireThrows<std::invalid_argument>(
            []
            {
                (void)SleepStageToClassLabel(
                    SleepStage::Unknown);
            },
            "Unknown must not convert to a class label.");

        RequireThrows<std::invalid_argument>(
            []
            {
                (void)SleepStageToClassLabel(
                    SleepStage::Movement);
            },
            "Movement must not convert to a class label.");

        RequireThrows<std::invalid_argument>(
            []
            {
                (void)ClassLabelToSleepStage(-1);
            },
            "Negative class labels must throw.");

        RequireThrows<std::invalid_argument>(
            []
            {
                (void)ClassLabelToSleepStage(5);
            },
            "Class labels above 4 must throw.");

        Pass("invalid class conversion");
    }

    void TestRoundTripConversion()
    {
        using eeg_to_hypnogram::SleepStage;

        const std::vector<SleepStage> stages{
            SleepStage::Wake,
            SleepStage::N1,
            SleepStage::N2,
            SleepStage::N3,
            SleepStage::Rem,
            SleepStage::Unknown,
            SleepStage::Movement,
        };

        for (const SleepStage expected : stages)
        {
            const std::string text =
                eeg_to_hypnogram::SleepStageToString(
                    expected);

            SleepStage parsed = SleepStage::Unknown;

            Require(
                eeg_to_hypnogram::TryParseSleepStage(
                    text,
                    parsed),
                "canonical string must round-trip parse.");

            Require(
                parsed == expected,
                "canonical string round-trip changed.");
        }

        Pass("round-trip conversion");
    }

} // namespace

int main()
{
    try
    {
        TestCanonicalStageParsing();
        TestSleepEdfAnnotationParsing();
        TestWhitespaceAndCaseNormalization();
        TestN4MergeRule();
        TestUnknownAndMovementParsing();
        TestNonStageAnnotationRejection();
        TestCanonicalStringConversion();
        TestTrainableStageRules();
        TestClassLabelConversion();
        TestInvalidClassConversion();
        TestRoundTripConversion();

        std::cout
            << "Sleep Stage tests passed\n"
            << "cases="
            << g_caseCount
            << '\n';

        return EXIT_SUCCESS;
    }
    catch (const std::exception &exception)
    {
        std::cerr
            << "sleep_stage_test failed: "
            << exception.what()
            << '\n';

        return EXIT_FAILURE;
    }
}
