#include "eeg_to_hypnogram/temporal_context.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    using FeatureMatrix =
        std::vector<std::vector<double>>;

    void Require(
        bool condition,
        const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error(message);
        }
    }

    void RequireMatrixEqual(
        const FeatureMatrix &actual,
        const FeatureMatrix &expected,
        const std::string &message)
    {
        if (actual != expected)
        {
            throw std::runtime_error(message);
        }
    }

    template <typename ExpectedException, typename Callable>
    void RequireThrows(
        Callable &&callable,
        const std::string &message)
    {
        bool threwExpectedException = false;

        try
        {
            callable();
        }
        catch (const ExpectedException &)
        {
            threwExpectedException = true;
        }
        catch (...)
        {
            throw std::runtime_error(
                message +
                " An unexpected exception type was thrown.");
        }

        Require(
            threwExpectedException,
            message);
    }

    void TestDefaultConfig()
    {
        const eeg_to_hypnogram::TemporalContextConfig config;

        Require(
            config.leftContext == 2,
            "Default left context must be 2.");

        Require(
            config.rightContext == 2,
            "Default right context must be 2.");
    }

    void TestEmptyInput()
    {
        const FeatureMatrix input;

        const eeg_to_hypnogram::TemporalContextConfig config;

        const auto output =
            eeg_to_hypnogram::BuildTemporalContextFeatures(
                input,
                config);

        Require(
            output.empty(),
            "Empty input must produce empty output.");
    }

    void TestEmptyInputWithInvalidConfig()
    {
        const FeatureMatrix input;

        eeg_to_hypnogram::TemporalContextConfig config;
        config.leftContext = -1;
        config.rightContext = -1;

        // 空输入在验证上下文配置之前直接返回。
        const auto output =
            eeg_to_hypnogram::BuildTemporalContextFeatures(
                input,
                config);

        Require(
            output.empty(),
            "Empty input with invalid context must preserve the old empty-output behavior.");
    }

    void TestNoContext()
    {
        const FeatureMatrix input = {
            {1.0, 10.0},
            {2.0, 20.0},
            {3.0, 30.0},
        };

        eeg_to_hypnogram::TemporalContextConfig config;
        config.leftContext = 0;
        config.rightContext = 0;

        const auto output =
            eeg_to_hypnogram::BuildTemporalContextFeatures(
                input,
                config);

        RequireMatrixEqual(
            output,
            input,
            "Zero temporal context must preserve the input matrix.");
    }

    void TestSymmetricContext()
    {
        const FeatureMatrix input = {
            {1.0, 10.0},
            {2.0, 20.0},
            {3.0, 30.0},
        };

        eeg_to_hypnogram::TemporalContextConfig config;
        config.leftContext = 1;
        config.rightContext = 1;

        const FeatureMatrix expected = {
            {
                1.0,
                10.0,
                1.0,
                10.0,
                2.0,
                20.0,
            },
            {
                1.0,
                10.0,
                2.0,
                20.0,
                3.0,
                30.0,
            },
            {
                2.0,
                20.0,
                3.0,
                30.0,
                3.0,
                30.0,
            },
        };

        const auto output =
            eeg_to_hypnogram::BuildTemporalContextFeatures(
                input,
                config);

        RequireMatrixEqual(
            output,
            expected,
            "Symmetric temporal context ordering or boundary replication is incorrect.");
    }

    void TestLeftOnlyContext()
    {
        const FeatureMatrix input = {
            {1.0},
            {2.0},
            {3.0},
        };

        eeg_to_hypnogram::TemporalContextConfig config;
        config.leftContext = 2;
        config.rightContext = 0;

        const FeatureMatrix expected = {
            {1.0, 1.0, 1.0},
            {1.0, 1.0, 2.0},
            {1.0, 2.0, 3.0},
        };

        const auto output =
            eeg_to_hypnogram::BuildTemporalContextFeatures(
                input,
                config);

        RequireMatrixEqual(
            output,
            expected,
            "Left-only temporal context is incorrect.");
    }

    void TestRightOnlyContext()
    {
        const FeatureMatrix input = {
            {1.0},
            {2.0},
            {3.0},
        };

        eeg_to_hypnogram::TemporalContextConfig config;
        config.leftContext = 0;
        config.rightContext = 2;

        const FeatureMatrix expected = {
            {1.0, 2.0, 3.0},
            {2.0, 3.0, 3.0},
            {3.0, 3.0, 3.0},
        };

        const auto output =
            eeg_to_hypnogram::BuildTemporalContextFeatures(
                input,
                config);

        RequireMatrixEqual(
            output,
            expected,
            "Right-only temporal context is incorrect.");
    }

    void TestSingleEpoch()
    {
        const FeatureMatrix input = {
            {1.0, 2.0},
        };

        const eeg_to_hypnogram::TemporalContextConfig config;

        const FeatureMatrix expected = {
            {
                1.0,
                2.0,
                1.0,
                2.0,
                1.0,
                2.0,
                1.0,
                2.0,
                1.0,
                2.0,
            },
        };

        const auto output =
            eeg_to_hypnogram::BuildTemporalContextFeatures(
                input,
                config);

        RequireMatrixEqual(
            output,
            expected,
            "A single epoch must be replicated across the full context window.");
    }

    void TestNegativeLeftContext()
    {
        const FeatureMatrix input = {
            {1.0},
        };

        eeg_to_hypnogram::TemporalContextConfig config;
        config.leftContext = -1;
        config.rightContext = 0;

        RequireThrows<std::invalid_argument>(
            [&]()
            {
                (void)eeg_to_hypnogram::
                    BuildTemporalContextFeatures(
                        input,
                        config);
            },
            "Negative left context must throw std::invalid_argument.");
    }

    void TestNegativeRightContext()
    {
        const FeatureMatrix input = {
            {1.0},
        };

        eeg_to_hypnogram::TemporalContextConfig config;
        config.leftContext = 0;
        config.rightContext = -1;

        RequireThrows<std::invalid_argument>(
            [&]()
            {
                (void)eeg_to_hypnogram::
                    BuildTemporalContextFeatures(
                        input,
                        config);
            },
            "Negative right context must throw std::invalid_argument.");
    }

    void TestInconsistentFeatureDimensions()
    {
        const FeatureMatrix input = {
            {1.0, 2.0},
            {3.0},
        };

        eeg_to_hypnogram::TemporalContextConfig config;
        config.leftContext = 0;
        config.rightContext = 0;

        RequireThrows<std::runtime_error>(
            [&]()
            {
                (void)eeg_to_hypnogram::
                    BuildTemporalContextFeatures(
                        input,
                        config);
            },
            "Inconsistent feature dimensions must throw std::runtime_error.");
    }

    void TestZeroDimensionalFeatures()
    {
        const FeatureMatrix input = {
            {},
            {},
        };

        eeg_to_hypnogram::TemporalContextConfig config;
        config.leftContext = 1;
        config.rightContext = 1;

        const FeatureMatrix expected = {
            {},
            {},
        };

        const auto output =
            eeg_to_hypnogram::BuildTemporalContextFeatures(
                input,
                config);

        RequireMatrixEqual(
            output,
            expected,
            "Consistent zero-dimensional feature rows must preserve the old behavior.");
    }

} // namespace

int main()
{
    try
    {
        TestDefaultConfig();
        TestEmptyInput();
        TestEmptyInputWithInvalidConfig();
        TestNoContext();
        TestSymmetricContext();
        TestLeftOnlyContext();
        TestRightOnlyContext();
        TestSingleEpoch();
        TestNegativeLeftContext();
        TestNegativeRightContext();
        TestInconsistentFeatureDimensions();
        TestZeroDimensionalFeatures();

        std::cout
            << "All temporal context tests passed.\n";

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Temporal context test failed: "
            << error.what()
            << '\n';

        return 1;
    }
}